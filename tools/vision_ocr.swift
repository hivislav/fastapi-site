// vision_ocr — распознавание текста в PDF и картинках средствами macOS.
//
// Зачем отдельный бинарь, а не Python-пакет: pyobjc-framework-Vision требует
// Python ≥3.10, а venv проекта — 3.9 (та же стена, что у sentence-transformers).
// Зато сам Vision в macOS есть всегда, работает ОФЛАЙН, понимает русский и не
// требует ни brew, ни tesseract. Поэтому здесь маленькая программа на Swift,
// которую собирает уже установленный swiftc (см. app/ai/rag_ocr.py).
//
// ЗАПУСК:  vision_ocr <файл> [языки] [предел страниц] [длинная сторона, px] [файл вывода]
//   пример: vision_ocr scan.pdf ru-RU,en-US 500 2400 /tmp/text.txt
//
// КУДА ПИШЕТСЯ ТЕКСТ. Если задан файл вывода (аргумент 5), распознанный текст
// идёт ТУДА, а не в stdout. Это не украшение: вызывающая сторона читает stderr
// (строки прогресса) построчно, и если текст лить в stdout, тот же процесс
// заполнит буфер канала (64 КБ — это 20–30 страниц скана) и ЗАВИСНЕТ, перестав
// отдавать прогресс. Именно так выглядела «застывшая» индексация, поэтому
// крупные данные хелпер пишет в файл, а канал оставляет для прогресса.
//
// ПОЧЕМУ ПАЧКАМИ И ПАРАЛЛЕЛЬНО: распознавание — самая дорогая часть индексации
// скана (на плотной странице A4 при 300 dpi это около секунды, на шумном скане
// больше). Vision умеет работать на нескольких ядрах, поэтому страницы идут
// ПАЧКАМИ: пачка рендерится последовательно (PDFKit не потокобезопасен) и
// распознаётся параллельно. Печать результатов идёт В ПОРЯДКЕ СТРАНИЦ, иначе
// текст скана перемешался бы, а прогресс показывал бы ерунду.
//
// ВЫВОД (важно для вызывающей стороны):
//   stdout — распознанный текст; страницы разделяются пустой строкой;
//   stderr — строки «PROGRESS <готово> <всего>» после каждой страницы (по ним
//            python-слой показывает прогресс в диалоге) и «ERROR <причина>».
//
// Код возврата: 0 — успех, 2 — нет аргументов, 3 — файл не открылся,
//               4 — Vision недоступен, 5 — пустой результат.

import Foundation
import Vision
import PDFKit
import ImageIO
import CoreGraphics

func fail(_ code: Int32, _ message: String) -> Never {
    FileHandle.standardError.write("ERROR \(message)\n".data(using: .utf8)!)
    exit(code)
}

func note(_ message: String) {
    FileHandle.standardError.write("\(message)\n".data(using: .utf8)!)
}

let arguments = CommandLine.arguments
guard arguments.count > 1 else { fail(2, "нужен путь к PDF или картинке") }
let path = arguments[1]
let wanted = arguments.count > 2 && !arguments[2].isEmpty
    ? arguments[2].split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }
    : ["ru-RU", "en-US"]
let pageLimit = arguments.count > 3 ? (Int(arguments[3]) ?? 0) : 0
// Длинная сторона страницы при рендере: 2400 px — разумный компромисс для текста,
// меньше — быстрее и грубее (мелкий шрифт распознаётся хуже).
let maxSide = arguments.count > 4 ? max(600, Int(arguments[4]) ?? 2400) : 2400
let outputPath = arguments.count > 5 ? arguments[5] : ""

// Вывод текста: в файл, если он задан (иначе — stdout, как в отладке).
var textOutput = FileHandle.standardOutput
if !outputPath.isEmpty {
    FileManager.default.createFile(atPath: outputPath, contents: nil)
    guard let handle = FileHandle(forWritingAtPath: outputPath) else {
        fail(6, "не удалось открыть файл вывода")
    }
    textOutput = handle
}
func emit(_ line: String) {
    textOutput.write((line + "\n").data(using: .utf8)!)
}

// Языки: Vision принимает не все — неподдерживаемые молча отбрасываем, иначе
// запрос падает целиком и распознавание не работает вовсе.
let supported = (try? VNRecognizeTextRequest.supportedRecognitionLanguages(
    for: .accurate, revision: VNRecognizeTextRequestRevision3)) ?? []
let languages = wanted.filter { supported.contains($0) }
let effective = languages.isEmpty ? supported.filter { $0.hasPrefix("en") } : languages
if languages.isEmpty && !wanted.isEmpty {
    note("WARN запрошенные языки не поддерживаются: \(wanted.joined(separator: ","))")
}

func makeRequest(correction: Bool = true) -> VNRecognizeTextRequest {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate          // медленнее, но заметно точнее
    request.usesLanguageCorrection = correction
    if !effective.isEmpty { request.recognitionLanguages = effective }
    return request
}

/// Рисует страницу PDF в картинку: Vision работает с изображением, а не с PDF.
/// Масштаб подбирается так, чтобы длинная сторона была ~2400 px: на 150 dpi
/// распознавание заметно хуже, а на 600 dpi — только медленнее.
func render(page: PDFPage) -> CGImage? {
    let box = page.bounds(for: .mediaBox)
    guard box.width > 1, box.height > 1 else { return nil }
    let target = CGFloat(maxSide)
    var scale = target / max(box.width, box.height)
    scale = min(max(scale, 1.5), 4.0)
    let width = Int(box.width * scale)
    let height = Int(box.height * scale)
    guard let context = CGContext(data: nil, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { return nil }
    context.scaleBy(x: scale, y: scale)
    context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: box.width, height: box.height))
    page.draw(with: .mediaBox, to: context)
    return context.makeImage()
}

/// Текст одной картинки в порядке чтения: Vision отдаёт наблюдения списком, и
/// без сортировки строки могут перемешаться (колонки, таблицы).
///
/// ГРАНИЦЫ АБЗАЦЕВ восстанавливаются по ВЕРТИКАЛЬНОМУ РАЗРЫВУ: если между
/// соседними строками промежуток заметно больше высоты строки, это новый абзац,
/// и в вывод уходит пустая строка. Без этого весь скан превращался бы в один
/// абзац: разбиение по структуре не нашло бы ни заголовков, ни разделов, а чанки
/// резались бы по счётчику символов посреди фразы.
/// Строка текста с координатами. Нужны для РАЗБОРА ВЁРСТКИ: по одним координатам
/// можно понять, что перед нами колонка, врезка или обычный абзац.
struct TextLine {
    let text: String
    let minX: CGFloat
    let maxX: CGFloat
    let top: CGFloat        // координаты в единицах страницы, ось Y вниз
    let bottom: CGFloat

    var width: CGFloat { maxX - minX }
    var height: CGFloat { bottom - top }
    var centerY: CGFloat { (top + bottom) / 2 }
    var centerX: CGFloat { (minX + maxX) / 2 }
}

func recognize(image: CGImage) -> [TextLine] {
    var lines = recognizeOnce(image: image, correction: true)
    if lines.isEmpty {
        // ПОВТОР БЕЗ ЯЗЫКОВОЙ КОРРЕКЦИИ. На плотных или однообразных страницах
        // (таблицы, повторяющиеся строки, шумный скан) корректор может вернуть
        // ПУСТОЙ результат — текста на странице при этом полно. Вторая попытка
        // без коррекции такие страницы вытаскивает; стоит она только на тех
        // страницах, где первая попытка ничего не дала.
        lines = recognizeOnce(image: image, correction: false)
        if !lines.isEmpty { note("WARN страница распознана без языковой коррекции") }
    }
    return lines
}

func recognizeOnce(image: CGImage, correction: Bool) -> [TextLine] {
    let request = makeRequest(correction: correction)
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    do {
        try handler.perform([request])
    } catch {
        note("WARN распознавание страницы не удалось: \(error.localizedDescription)")
        return []
    }
    let recognized = (request.results ?? []).compactMap { observation -> TextLine? in
        guard let text = observation.topCandidates(1).first?.string else { return nil }
        let box = observation.boundingBox
        return TextLine(text: text,
                        minX: box.minX,
                        maxX: box.maxX,
                        top: 1 - box.maxY,      // Vision считает Y снизу, нам удобнее сверху
                        bottom: 1 - box.minY)
    }
    return layoutBlocks(recognized).flatMap { $0 }
}

// ---------------------------------------------------------------------------
// РАЗБОР ВЁРСТКИ: колонки, врезки, порядок чтения
// ---------------------------------------------------------------------------
// Простой сортировки «сверху вниз» для комиксов, газетных вырезок и книг правил
// НЕ ХВАТАЕТ: текст идёт колонками и отдельными окнами, и построчная сортировка
// перемешивает их — «АЛЬФА, ДЕЛЬТА, БРАВО, ЭХО» вместо «вся левая колонка, потом
// правая». Поэтому страница режется по ПУСТОМУ МЕСТУ между блоками (классический
// XY-cut): сначала ищется вертикальная полоса без текста (это стык колонок),
// затем горизонтальная (это граница абзаца, заголовка или врезки), и так
// рекурсивно. Порядок частей при этом и есть порядок чтения.

/// Медианная высота строки: с ней сравниваются разрывы, чтобы пороги не зависели
/// от кегля страницы.
func medianLineHeight(_ lines: [TextLine]) -> CGFloat {
    let heights = lines.map { $0.height }.sorted()
    guard !heights.isEmpty else { return 0.012 }
    return max(0.004, heights[heights.count / 2])
}

/// Самая широкая вертикальная полоса БЕЗ текста. Возвращает её середину по X.
/// Пустая полоса через всю область — это стык колонок; разрыв между словами
/// внутри строки такой полосы не даёт.
func widestColumnGap(_ lines: [TextLine], minGap: CGFloat) -> CGFloat? {
    let sorted = lines.sorted { $0.minX < $1.minX }
    var rightEdge = sorted[0].maxX
    var best: (gap: CGFloat, at: CGFloat)? = nil
    for line in sorted.dropFirst() {
        let gap = line.minX - rightEdge
        if gap >= minGap, best == nil || gap > best!.gap {
            best = (gap, rightEdge + gap / 2)
        }
        rightEdge = max(rightEdge, line.maxX)
    }
    return best?.at
}

/// Самая широкая горизонтальная полоса без текста — граница блоков по вертикали.
///
/// ПОРОГ — ДОЛЯ ВЫСОТЫ СТРОКИ (0.3), а не «самый большой разрыв страницы»:
/// попытка считать порог от распределения разрывов ломается, когда границ на
/// странице много (в документе с пропусками между абзацами «обычным» разрывом
/// оказывается как раз граница, и тогда не разрезается ничего). Междустрочный
/// интервал при этом много меньше 0.3 высоты, а разрыв перед абзацем, заголовком
/// или врезкой — больше, поэтому граница проходит между ними.
///
/// Разрезать лишний раз НЕ СТРАШНО: разбиение по структуре всё равно собирает
/// раздел из заголовка и следующих за ним абзацев, а вот склеить разные разделы —
/// значит потерять адрес фрагмента. Поэтому порог смещён в сторону разрезов.
func widestRowGap(_ lines: [TextLine], minGap: CGFloat) -> CGFloat? {
    let sorted = lines.sorted { $0.top < $1.top }
    guard sorted.count >= 3 else { return nil }
    var gaps: [(gap: CGFloat, at: CGFloat)] = []
    var bottomEdge = sorted[0].bottom
    for line in sorted.dropFirst() {
        let gap = line.top - bottomEdge
        gaps.append((max(0, gap), (line.top + bottomEdge) / 2))
        bottomEdge = max(bottomEdge, line.bottom)
    }
    let threshold = max(minGap, medianLineHeight(lines) * 0.3)
    return gaps.filter { $0.gap >= threshold }.max { $0.gap < $1.gap }?.at
}

/// Режет страницу на блоки в порядке чтения.
func layoutBlocks(_ lines: [TextLine], depth: Int = 0) -> [[TextLine]] {
    if lines.count <= 1 || depth > 12 {
        return lines.isEmpty ? [] : [lines]
    }
    let height = medianLineHeight(lines)
    // Стык колонок: пустая полоса шириной хотя бы в полторы строки.
    if let cut = widestColumnGap(lines, minGap: max(0.012, height * 1.5)) {
        let left = lines.filter { $0.centerX < cut }
        let right = lines.filter { $0.centerX >= cut }
        if !left.isEmpty && !right.isEmpty {
            return layoutBlocks(left, depth: depth + 1)
                 + layoutBlocks(right, depth: depth + 1)
        }
    }
    // Граница блоков по вертикали: обычный междустрочный интервал не разрезает,
    // а пропуск перед абзацем, заголовком или врезкой — разрезает.
    if let cut = widestRowGap(lines, minGap: 0.0028) {
        let top = lines.filter { $0.centerY < cut }
        let bottom = lines.filter { $0.centerY >= cut }
        if !top.isEmpty && !bottom.isEmpty {
            return layoutBlocks(top, depth: depth + 1)
                 + layoutBlocks(bottom, depth: depth + 1)
        }
    }
    return [lines.sorted { $0.top < $1.top }]
}

/// Абзацы страницы: блоки в порядке чтения, внутри блока строки сверху вниз.
/// Заголовок отделяется от своего абзаца РАЗРЫВОМ (см. widestRowGap), а не
/// высотой строки: по высоте рамки Vision заголовок капсом неотличим от строки
/// строчными с выносными элементами.
func pageParagraphs(_ lines: [TextLine]) -> [[String]] {
    return layoutBlocks(lines).map { block in
        block.sorted { $0.top < $1.top }.map { $0.text }
    }
}


// Источник: PDF или одна картинка. Картинку тоже умеем — одиночный скан в JPG
// встречается не реже, чем многостраничный PDF.
//
// ФОРМАТ ОПРЕДЕЛЯЕТСЯ ПО СОДЕРЖИМОМУ, а не по расширению. Файлы приходят из
// потоковой загрузки под именем вида «xxxx.part», и проверка по расширению
// отправляла настоящий PDF в ветку картинок («картинка не открылась») — живой
// случай, который поймал прогон через приложение.
func looksLikePDF(_ path: String) -> Bool {
    guard let handle = FileHandle(forReadingAtPath: path) else { return false }
    defer { try? handle.close() }
    let head = handle.readData(ofLength: 4)
    return head == Data([0x25, 0x50, 0x44, 0x46])          // «%PDF»
}

let debugEnabled = ProcessInfo.processInfo.environment["VISION_OCR_DEBUG"] == "1"

let isPDF = looksLikePDF(path)
var document: PDFDocument? = nil
var single: CGImage? = nil
var total = 0

if isPDF {
    guard let opened = PDFDocument(url: URL(fileURLWithPath: path)) else {
        fail(3, "PDF не открылся")
    }
    document = opened
    total = pageLimit > 0 ? min(pageLimit, opened.pageCount) : opened.pageCount
} else {
    guard let source = CGImageSourceCreateWithURL(
        URL(fileURLWithPath: path) as CFURL, nil),
        let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        fail(3, "картинка не открылась")
    }
    single = image
    total = 1
}

guard total > 0 else { fail(5, "не удалось получить ни одной страницы") }

/// Страницы обрабатываются ПОСЛЕДОВАТЕЛЬНО, и это осознанно: замер на плотной
/// странице A4 (300 dpi) дал 0.59 с/страницу и в параллельном, и в
/// последовательном варианте — Vision сам загружает все ядра, поэтому пачки
/// ничего не ускоряли, зато добавляли гонку при записи результатов из потоков.
/// Один язык за раз — тоже осознанно: так текст страницы не перемешивается.
var written = 0
for number in 0..<total {
    autoreleasepool {
        let renderStarted = Date()
        let frame = document?.page(at: number).flatMap { render(page: $0) } ?? single
        if debugEnabled {
            note(String(format: "DEBUG страница %d: рендер %.0f мс",
                        number + 1, Date().timeIntervalSince(renderStarted) * 1000))
            note("DEBUG языки: запрошены [\(wanted.joined(separator: ","))] | "
                 + "используются [\(effective.joined(separator: ","))]")
        }
        guard let image = frame else {
            note("PROGRESS \(number + 1) \(total)")
            return
        }
        let started = Date()
        let lines = recognize(image: image)
        // Печатаем БЛОКАМИ: строки одного блока — своими строками, между блоками
        // пустая строка. Так вызывающая сторона видит абзацы и колонки, а не
        // сплошную кашу из строк, перемешанных по всей странице.
        let paragraphs = pageParagraphs(lines)
        if debugEnabled {
            note(String(format: "DEBUG страница %d: распознавание %.0f мс, строк %d, блоков %d",
                        number + 1, Date().timeIntervalSince(started) * 1000,
                        lines.count, paragraphs.count))
        }
        for paragraph in paragraphs {
            for line in paragraph {
                emit(line)
                written += 1
            }
            emit("")                                // граница блока (абзаца)
        }
        if debugEnabled {
            // Координаты строк: по ним видно, почему блоки разрезаны именно так.
            for line in lines.sorted(by: { $0.top < $1.top }) {
                note(String(format: "DEBUG строка top=%.4f h=%.4f x=%.3f-%.3f | %@",
                            line.top, line.height, line.minX, line.maxX,
                            String(line.text.prefix(34))))
            }
        }
        emit("")                                    // разделитель страниц
        note("PROGRESS \(number + 1) \(total)")
    }
}

if written == 0 { fail(5, "текст не распознан ни на одной странице") }
if !outputPath.isEmpty { try? textOutput.close() }
