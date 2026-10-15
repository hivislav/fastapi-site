"""Проверка MCP-сервера «Поддержка» (клиенты и обращения) и правила команды `/help`.

Запуск:

    ./venv/bin/python tools/check_support_mcp.py
    ./venv/bin/python tools/check_support_mcp.py --keep   # не удалять временный файл данных

ЧТО ПРОВЕРЯЕТСЯ.

  [1] СЕРВЕР (живой, нужен Node): рукопожатие и `tools/list` — семь
      инструментов; работоспособность каждого на ВРЕМЕННОМ файле данных
      (`TICKETS_FILE`), чтобы проверка не трогала ни демонстрационные, ни рабочие
      данные проекта:
        * `search_tickets` по вопросу «почему не работает авторизация?» находит
          ОБРАЩЕНИЕ ПРО ЭТО ЖЕ — и первым в списке;
        * `get_ticket` отдаёт статус, переписку и клиента (то, из чего
          складывается ответ «с учётом данных тикета»);
        * номер без префикса («1041») разрешается, а несуществующий — честный
          отказ с перечислением того, что есть (а не выдуманное обращение);
        * `create_ticket` ЗАВОДИТ обращение: номер следующий по максимуму,
          заявитель привязывается к существующей карточке клиента, а неизвестный
          клиент получает новую карточку; ПОХОЖЕЕ ОТКРЫТОЕ обращение заводить
          вторым сервер отказывается, пока не подтвердят (`force`) — закрытые
          дубликатом не считаются; вызов без заявителя отвергает схема;
        * `add_comment` пишет в переписку, и запись видна следующему чтению;
        * ИСПОРЧЕННЫЙ файл данных не перезаписывается: запись отменяется с
          причиной, содержимое файла остаётся прежним.

  [2] ПРАВИЛО КОМАНДЫ `/help` (без сети и без Node, как в tools/check_dev_help.py):
      когда команда идёт в поддержку (app/ai/support.py) и когда НЕ идёт —
      вопросы об устройстве проекта данных обращения не требуют; номер обращения
      в вопросе читается с заглавными буквами («TCK-1041», но не «fix-42»);
      выключенный сервер называется причиной, а не молчанием.

  [3] ЗАЯВИТЕЛЬ — ИЗ ПРОФИЛЯ, А НЕ ОТ МОДЕЛИ (без сети и Node): дополнитель
      аргументов (`chat._fill_requester`) подставляет имя клиента из профиля в
      вызов `create_ticket`; заглушки модели («клиент», «не указан») считаются
      пустым значением и заменяются; УЖЕ НАЗВАННЫЙ человеком заявитель не
      перебивается; чужие серверы и инструменты не трогаются; пустое имя в
      профиле не подставляется вовсе. Там же — что дополнитель ЗАРЕГИСТРИРОВАН и
      что сбой одного дополнителя не ломает остальные вызовы.

Код возврата: 0 — всё сошлось, 1 — есть отказы. Рабочие данные проекта не
трогаются: файл обращений задаётся переменной `TICKETS_FILE`, а workspace и
профили — во временном каталоге (`AGENT_*_FILE` выставляются ДО импорта).
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Изоляция данных: всё пишем во временный каталог ------------------------
# Профиль читается проверкой заявителя, workspace — импортом веб-слоя; ни то, ни
# другое не имеет права коснуться рабочих файлов проекта.
_TMP = tempfile.mkdtemp(prefix="support-check-")
os.environ["AGENT_WORKSPACE_FILE"] = os.path.join(_TMP, "workspace.json")
os.environ["AGENT_MEMORY_FILE"] = os.path.join(_TMP, "agent_memory.json")
os.environ["AGENT_PROFILES_FILE"] = os.path.join(_TMP, "profiles.json")
os.environ["RAG_DIR"] = os.path.join(_TMP, "rag")
os.environ["MCP_ATTACH_DIR"] = os.path.join(_TMP, "mcp_files")
os.environ["RAG_EMBED_BACKEND"] = "hashing"
os.environ["RAG_RERANK_BACKEND"] = "features"
os.environ["PERIODIC_ENABLED"] = "0"

from app.ai import dev_help  # noqa: E402
from app.ai import client as llm_client  # noqa: E402
from app.ai import mcp as mcp_store  # noqa: E402
from app.ai import profiles as profile_store  # noqa: E402
from app.ai import support  # noqa: E402
from app.ai import workspace as workspace_store  # noqa: E402
from app.routers import chat  # noqa: E402
from app.schemas import ChatMessage  # noqa: E402

FAILED = []
# Файл данных для проверки: свой у каждого запуска, чтобы рабочие данные проекта
# (data/tickets.json) и образец в репозитории остались нетронутыми.
SANDBOX = ""


def check(name, ok, detail=""):
    """Печать результата проверки в стиле остальных tools/check_*.py."""
    print(("  ok   " if ok else "  FAIL ") + str(name)
          + ("" if ok or not detail else f" — {detail}"))
    if not ok:
        FAILED.append(str(name))
    return bool(ok)


def call(tool, **arguments):
    """Живой вызов инструмента сервера поддержки (через клиент приложения)."""
    result = mcp_store.run_calls([
        {"server": mcp_store.SUPPORT, "tool": tool, "arguments": arguments},
    ])[0]
    return result


def section(title):
    print("\n" + title)


# ---------------------------------------------------------------------------
# [1] Сервер: рукопожатие, инструменты, чтения и запись
# ---------------------------------------------------------------------------
def check_server():
    section("[1] Сервер поддержки: рукопожатие и tools/list")
    found = mcp_store.discover(mcp_store.SUPPORT, force=True)
    if not check("сервер ответил на initialize и tools/list", found.get("ok"),
                 str(found.get("error") or "")[:200]):
        return
    names = [str(item.get("name") or "") for item in (found.get("tools") or [])]
    print(f"  --   сервер ответил как «{found.get('server_name')}», "
          f"инструментов: {len(names)}")
    for name in names:
        print(f"       · {name}")
    expected = ["list_users", "get_user", "search_tickets", "get_ticket",
                "list_tickets", "create_ticket", "add_comment"]
    check("объявлены все семь инструментов", sorted(names) == sorted(expected),
          f"объявлено: {sorted(names)}")

    section("[1а] Файл данных: создаётся из образца репозитория")
    store_file = os.environ["TICKETS_FILE"]
    check("рабочего файла данных ещё нет", not os.path.exists(store_file))
    first = call("list_users", query="")
    check("первый вызов создал файл данных", os.path.exists(store_file),
          str(first.get("error") or "")[:200])
    check("в созданном файле есть клиенты и обращения", first.get("ok")
          and "КЛИЕНТЫ ПОДДЕРЖКИ (3)" in first.get("text", ""),
          str(first.get("error") or first.get("text") or "")[:200])
    # Схема инструмента — это тоже защита: предел списка объявлен на сервере, и
    # выход за него отвергает сам протокол, а не «тихо обрезает» клиент.
    too_big = call("list_users", limit=50)
    check("выход за предел схемы отвергнут протоколом",
          not too_big.get("ok") and "limit" in str(too_big.get("error") or ""),
          str(too_big.get("error") or "")[:200])

    section("[1б] Поиск обращения по вопросу клиента (без номера)")
    hit = call("search_tickets", query="не работает вход по паролю")
    text = str(hit.get("text") or "")
    check("поиск по вопросу заявителя нашёл обращение", hit.get("ok"), text[:200])
    check("первым идёт обращение про авторизацию", text.splitlines()[1:2]
          and "TCK-1041" in text.splitlines()[1], text.splitlines()[1:2])
    check("в найденном обращении назван клиент и статус",
          "Вячеслав" in text and "в работе" in text, text[:200])
    noise = call("search_tickets", query="локальная модель не отвечает")
    noise_text = str(noise.get("text") or "")
    check("шум от служебных слов не подмешивается (одно обращение)",
          noise.get("ok") and "НАЙДЕННЫЕ ОБРАЩЕНИЯ (1 " in noise_text,
          noise_text.splitlines()[:1])
    empty = call("search_tickets", query="зелёный квазар в бухгалтерии")
    check("нет совпадений — отказ с перечнем того, что есть",
          not empty.get("ok") and "В базе 6 обращений" in str(empty.get("error") or ""),
          str(empty.get("error") or "")[:200])

    section("[1в] Обращение целиком: статус, переписка, клиент")
    full = call("get_ticket", id="TCK-1041")
    full_text = str(full.get("text") or "")
    check("обращение прочитано по номеру", full.get("ok"), full_text[:200])
    for needle in ("Статус: в работе", "Клиент: Вячеслав (u-1001)", "ПЕРЕПИСКА (",
                   "поддержка (2026-10-14 10:15)"):
        check("в обращении есть «%s»" % needle, needle in full_text, full_text[:200])
    short = call("get_ticket", id="1041")
    check("номер без префикса разрешается в то же обращение",
          short.get("ok") and "TCK-1041" in str(short.get("text") or ""),
          str(short.get("error") or "")[:200])
    missing = call("get_ticket", id="TCK-9999")
    check("несуществующий номер — отказ с причиной, а не выдуманное обращение",
          not missing.get("ok") and "нет" in str(missing.get("error") or ""),
          str(missing.get("error") or "")[:200])

    section("[1г] Клиенты и список обращений")
    user = call("get_user", user="Анна")
    check("карточка клиента по имени", user.get("ok")
          and "Имя: Анна" in str(user.get("text") or "")
          and "u-1002" in str(user.get("text") or ""),
          str(user.get("error") or "")[:200])
    unknown_user = call("get_user", user="Гарри Поттер")
    check("неизвестный клиент — отказ с именами известных",
          not unknown_user.get("ok") and "Известные клиенты" in str(unknown_user.get("error") or ""),
          str(unknown_user.get("error") or "")[:200])
    listed = call("list_tickets", status="новый")
    check("список обращений по статусу", listed.get("ok")
          and "TCK-1028" in str(listed.get("text") or ""),
          str(listed.get("error") or "")[:200])

    section("[1д] Запись: комментарий в переписку")
    before = call("get_ticket", id="TCK-1028")
    before_count = str(before.get("text") or "").count("  · ")
    wrote = call("add_comment", id="TCK-1028", text="Проверка записи.", author="проверка")
    check("комментарий записан", wrote.get("ok")
          and "СООБЩЕНИЕ ЗАПИСАНО" in str(wrote.get("text") or ""),
          str(wrote.get("error") or "")[:200])
    after = call("get_ticket", id="TCK-1028")
    after_text = str(after.get("text") or "")
    check("запись видна следующему чтению",
          "Проверка записи." in after_text
          and str(after_text).count("  · ") == before_count + 1,
          after_text[:200])
    no_ticket = call("add_comment", id="TCK-0001", text="В никуда.")
    check("запись в несуществующее обращение отклонена",
          not no_ticket.get("ok") and "не записан" in str(no_ticket.get("error") or ""),
          str(no_ticket.get("error") or "")[:200])

    section("[1ж] Создание обращения: номер, заявитель, дубликаты")
    created = call("create_ticket", requester="Вячеслав",
                   subject="Медленно открывается список проектов",
                   description="Список проектов открывается несколько секунд, хотя проектов немного.",
                   product="интерфейс (app/web/chat.html)")
    created_text = str(created.get("text") or "")
    check("обращение создано", created.get("ok") and "ОБРАЩЕНИЕ СОЗДАНО" in created_text,
          str(created.get("error") or "")[:200])
    check("номер — следующий за максимальным (в образце максимум TCK-1042)",
          "TCK-1043" in created_text, created_text.splitlines()[:1])
    check("заявитель привязан к СУЩЕСТВУЮЩЕЙ карточке клиента",
          "(u-1001)" in created_text and "существующему клиенту" in created_text,
          created_text[:300])
    stranger = call("create_ticket", requester="Новый Клиент",
                    subject="Пропал доступ к выгрузкам",
                    description="Раньше мог выгружать отчёты, теперь пишет, что прав нет.")
    stranger_text = str(stranger.get("text") or "")
    check("неизвестному клиенту заводится карточка",
          stranger.get("ok") and "карточка заведена" in stranger_text
          and "(u-1004)" in stranger_text, stranger_text[:300])
    later = call("list_users", query="Новый Клиент")
    check("новая карточка видна в списке клиентов",
          later.get("ok") and "Новый Клиент" in str(later.get("text") or ""),
          str(later.get("error") or "")[:200])
    check("созданное обращение читается поиском",
          (lambda res: res.get("ok") and "TCK-1043" in str(res.get("text") or ""))(
              call("search_tickets", query="медленно открывается список проектов")),
          "поиск по теме созданного обращения")
    twin = call("create_ticket", requester="Вячеслав",
                subject="Опять не работает вход по паролю",
                description="Снова «Неверный пароль» на телефоне.")
    twin_error = str(twin.get("error") or "")
    check("похожее ОТКРЫТОЕ обращение — отказ, а не молчаливый дубликат",
          not twin.get("ok") and "TCK-1041" in twin_error and "force=true" in twin_error,
          twin_error[:200])
    forced = call("create_ticket", requester="Вячеслав",
                  subject="Опять не работает вход по паролю",
                  description="Снова «Неверный пароль» на телефоне.", force=True)
    check("после подтверждения (force) обращение создаётся",
          forced.get("ok") and "TCK-1045" in str(forced.get("text") or ""),
          str(forced.get("error") or "")[:200])
    # Заявитель — НЕобязательный аргумент схемы (его подставляет код из профиля),
    # но пустое значение сервер отвергает по смыслу: обращение «от никого» хуже
    # честного отказа. Обязательный аргумент, которого модель не знает, заставлял
    # диспетчера вообще не вызывать инструмент (живой случай 15.10).
    no_requester = call("create_ticket", subject="Без заявителя")
    check("вызов без заявителя отклонён по смыслу, с причиной",
          not no_requester.get("ok")
          and "Заявитель не назван" in str(no_requester.get("error") or ""),
          str(no_requester.get("error") or "")[:200])
    blank = call("create_ticket", requester="   ", subject="Пустой заявитель")
    check("заявитель из одних пробелов тоже не проходит",
          not blank.get("ok") and "Заявитель не назван" in str(blank.get("error") or ""),
          str(blank.get("error") or "")[:200])

    section("[1з] Значения-синонимы: «high» и «in progress» понимаются")
    english = call("create_ticket", requester="Новый Клиент",
                   subject="Проверка английского приоритета",
                   description="Обращение для проверки нормализации значений.",
                   priority="high")
    check("приоритет «high» принят и переведён в «высокий»",
          english.get("ok") and "Приоритет: высокий" in str(english.get("text") or ""),
          str(english.get("error") or "")[:200])
    wrong_priority = call("create_ticket", requester="Новый Клиент",
                          subject="Проверка нераспознанного приоритета",
                          description="Проверка отказа на неизвестном значении.",
                          priority="когда-нибудь")
    check("нераспознанный приоритет — отказ со списком допустимых",
          not wrong_priority.get("ok")
          and "Допустимые" in str(wrong_priority.get("error") or ""),
          str(wrong_priority.get("error") or "")[:200])
    english_status = call("search_tickets", query="не работает авторизация",
                          status="in progress")
    check("статус «in progress» понят как «в работе»",
          english_status.get("ok") and "TCK-1041" in str(english_status.get("text") or ""),
          str(english_status.get("error") or "")[:200])
    wrong_status = call("list_tickets", status="когда-нибудь")
    check("нераспознанный статус — отказ со списком известных и групп",
          not wrong_status.get("ok")
          and "Известные статусы" in str(wrong_status.get("error") or "")
          and "открытые" in str(wrong_status.get("error") or ""),
          str(wrong_status.get("error") or "")[:200])
    # ГРУППЫ СТАТУСОВ: «открыто/открытые/активные» — это не один статус, а всё,
    # что ещё не закончено. Живой случай 14.10: на «/support — что у нас в работе»
    # диспетчер прислал статус «открыто», сервер отклонил его, и режим вместо
    # списка просил человека выбрать статус.
    for word in ("открыто", "открытые", "активные"):
        group = call("list_tickets", status=word)
        text_group = str(group.get("text") or "")
        check("группа «%s» находит и «в работе», и «новый»" % word,
              group.get("ok") and "TCK-1041" in text_group and "TCK-1028" in text_group
              and "TCK-1019" not in text_group,
              str(group.get("error") or text_group)[:200])
    closed_group = call("list_tickets", status="закрытые")
    check("группа «закрытые» находит и «решён», и «закрыт»",
          closed_group.get("ok") and "TCK-1042" in str(closed_group.get("text") or "")
          and "TCK-1019" in str(closed_group.get("text") or ""),
          str(closed_group.get("error") or "")[:200])
    one_status = call("list_tickets", status="в работе")
    check("точный статус по-прежнему работает отдельно",
          one_status.get("ok") and "TCK-1028" not in str(one_status.get("text") or ""),
          str(one_status.get("error") or "")[:200])

    section("[1и] Чужое обращение не отдаётся (проверка на сервере)")
    alien = call("get_ticket", id="TCK-1042", client="Вячеслав")
    check("обращение другого клиента — отказ с причиной",
          not alien.get("ok")
          and "принадлежит другому клиенту" in str(alien.get("error") or "")
          and "Анна" in str(alien.get("error") or ""),
          str(alien.get("error") or "")[:250])
    mine = call("get_ticket", id="TCK-1041", client="Вячеслав")
    check("своё обращение читается с тем же аргументом", mine.get("ok")
          and "Клиент: Вячеслав" in str(mine.get("text") or ""),
          str(mine.get("error") or "")[:200])
    without_client = call("get_ticket", id="TCK-1042")
    check("без аргумента клиента обращение читается (проверки и диагностика)",
          without_client.get("ok"), str(without_client.get("error") or "")[:200])
    own_list = call("list_tickets", user="Вячеслав", status="open")
    # У Анны открытых обращений нет (оба закрыты) — берём группу «закрытые»,
    # иначе проверка ловила бы пустоту и падала по неверной причине.
    other_list = call("list_tickets", user="Анна", status="закрытые")
    check("список с фильтром по клиенту отдаёт только его обращения",
          own_list.get("ok") and "Вячеслав" in str(own_list.get("text") or "")
          and "Анна" not in str(own_list.get("text") or ""),
          str(own_list.get("text") or "")[:200])
    check("у другого клиента свой список",
          other_list.get("ok") and "Анна" in str(other_list.get("text") or "")
          and "Вячеслав" not in str(other_list.get("text") or ""),
          str(other_list.get("text") or "")[:200])

    section("[1е] Испорченный файл данных не перезаписывается")
    broken = os.path.join(SANDBOX, "broken.json")
    with open(broken, "w", encoding="utf-8") as handle:
        handle.write("{это не JSON")
    os.environ["TICKETS_FILE"] = broken
    read_broken = call("list_tickets")
    check("чтение испорченного файла — отказ с причиной",
          not read_broken.get("ok")
          and "не смог получить данные" in str(read_broken.get("error") or ""),
          str(read_broken.get("error") or "")[:200])
    write_broken = call("add_comment", id="TCK-1028", text="Не должно записаться.")
    with open(broken, encoding="utf-8") as handle:
        body = handle.read()
    check("запись в испорченный файл отменена, файл не тронут",
          not write_broken.get("ok") and body == "{это не JSON", body[:80])
    os.environ["TICKETS_FILE"] = store_file


# ---------------------------------------------------------------------------
# [2] Правила команды /support: разбор, отсечки и промпт (без сети и Node)
# ---------------------------------------------------------------------------
def check_command_rules():
    section("[2] Команда /support: разбор, признаки и отсечки вызовов")
    check("команда узнаётся, а «/supporter» — нет",
          support.is_command("/support TCK-1041 вопрос") and support.is_command("/support")
          and not support.is_command("/supporter")
          and not support.is_command("/support-me"))
    ok, ticket, asked = support.parse("/support TCK-1041 почему не работает вход?")
    check("номер обращения берётся из команды, вопрос — из остатка",
          ok and ticket == "TCK-1041" and asked == "почему не работает вход?",
          "%r %r %r" % (ok, ticket, asked))
    _, ticket_only, empty_question = support.parse("/support TCK-1041")
    check("один номер без вопроса — законная форма (разбор обращения)",
          ticket_only == "TCK-1041" and empty_question == ""
          and support.question("/support TCK-1041") == support.DEFAULT_QUESTION)
    _, no_ticket, some_question = support.parse("/support заведи обращение по жалобе")
    check("без номера вопрос уходит целиком",
          no_ticket == "" and some_question == "заведи обращение по жалобе",
          "%r %r" % (no_ticket, some_question))
    check("номер без префикса принимается, имя ветки и дата — нет",
          support.ticket_id("1041 не работает") == "1041"
          and support.ticket_id("TCK-1041 x") == "TCK-1041"
          and support.ticket_id("fix-42 x") == ""
          and support.ticket_id("2026-10-14 x") == "")
    check("номер ищется в ПЕРВОМ слове, а не в упоминании внутри вопроса",
          support.parse("/support TCK-1041 а что с TCK-9999?")[1] == "TCK-1041")
    check("чтение выбранного обращения задаёт КОД (один вызов)",
          support.fetch_calls("TCK-1041")
          == [{"server": support.SERVER_ID, "tool": support.TICKET_TOOL,
               "arguments": {"id": "TCK-1041"}}]
          and support.fetch_calls("") == [])

    section("[2а] Изменения — только по прямой просьбе человека")
    check("просьбу завести обращение видно",
          support.writes_allowed("заведи обращение по жалобе")
          and support.writes_allowed("создай тикет на выгрузку")
          and support.writes_allowed("ответь клиенту, что делать"))
    check("обычный вопрос просьбой не считается",
          not support.writes_allowed("что с обращением TCK-1041?")
          and not support.writes_allowed("почему не работает вход"))
    calls = [
        {"server": support.SERVER_ID, "tool": "search_tickets", "arguments": {}},
        {"server": support.SERVER_ID, "tool": "create_ticket", "arguments": {}},
        {"server": mcp_store.GIT, "tool": "git_status", "arguments": {}},
    ]
    kept, notes = support.filter_calls(calls, allow_writes=False)
    check("без просьбы остаются только чтения поддержки",
          [c["tool"] for c in kept] == ["search_tickets"], json.dumps(kept, ensure_ascii=False))
    check("отброшенное названо словами (и изменение, и чужой сервер)",
          any("изменение" in note for note in notes)
          and any("только сервер" in note for note in notes),
          " | ".join(notes)[:300])
    kept_writes, _ = support.filter_calls(calls, allow_writes=True)
    check("с просьбой изменение проходит, чужой сервер — нет",
          [c["tool"] for c in kept_writes] == ["search_tickets", "create_ticket"],
          json.dumps(kept_writes, ensure_ascii=False))
    # ИЗОЛЯЦИЯ КЛИЕНТА: поиск и списки получают фильтр, а инструменты «про
    # клиентов вообще» отбрасываются — иначе в ответ попадают ЧУЖИЕ обращения
    # (живой случай 15.10: «не получается найти файл через rag» показал TCK-1039
    # Дмитрия и TCK-1042 Анны).
    scoped, scoped_notes = support.scope_calls(
        [{"server": support.SERVER_ID, "tool": support.SEARCH_TOOL,
          "arguments": {"query": "файл"}},
         {"server": support.SERVER_ID, "tool": "list_users", "arguments": {}},
         {"server": support.SERVER_ID, "tool": "get_user",
          "arguments": {"user": "Анна"}}],
        "Вячеслав")
    check("поиск и списки сужаются текущим клиентом",
          scoped and scoped[0]["arguments"].get("user") == "Вячеслав"
          and len(scoped) == 1, json.dumps(scoped, ensure_ascii=False))
    check("инструменты про клиентов вообще отброшены с пояснением",
          len(scoped_notes) == 2 and all("текущего клиента" in note for note in scoped_notes),
          " | ".join(scoped_notes)[:200])
    check("без клиента (пустой профиль) ограничивать нечем — вызовы не меняются",
          support.scope_calls(
              [{"server": support.SERVER_ID, "tool": "list_users", "arguments": {}}],
              "")[0]
          == [{"server": support.SERVER_ID, "tool": "list_users", "arguments": {}}])
    check("выбранное обращение читается с проверкой принадлежности",
          support.fetch_calls("TCK-1041", "Вячеслав")
          == [{"server": support.SERVER_ID, "tool": support.TICKET_TOOL,
               "arguments": {"id": "TCK-1041", "client": "Вячеслав"}}]
          and support.fetch_calls("TCK-1041") == [{"server": support.SERVER_ID,
                                                   "tool": support.TICKET_TOOL,
                                                   "arguments": {"id": "TCK-1041"}}],
          json.dumps(support.fetch_calls("TCK-1041", "Вячеслав"), ensure_ascii=False))
    check("вызовов режима не больше предела",
          len(support.filter_calls([dict(calls[0]) for _ in range(12)])[0])
          <= support.MAX_MODE_CALLS)

    section("[2б] Промпт собирается по данным")
    with_ticket = support.prompt_for(True, True)
    without_ticket = support.prompt_for(False, False)
    check("с обращением правило опирается на переписку",
          "ДАННЫЕ ОБРАЩЕНИЯ — ФАКТ" in with_ticket)
    check("без обращения правило требует назвать номер, а не выдумывать",
          "ВЫБРАННОЕ ОБРАЩЕНИЕ В ЭТОМ ОТВЕТЕ НЕ ПРИВЕДЕНО" in without_ticket
          and support.COMMAND in without_ticket)
    check("без фрагментов правило про ссылки отменено",
          "ФРАГМЕНТОВ ДОКУМЕНТАЦИИ В ЭТОМ ОТВЕТЕ НЕТ" in without_ticket
          and "[1]" in support.prompt_for(True, True))
    messages = support.messages("почему не работает вход?", ticket_block="ОБРАЩЕНИЕ TCK-1041")
    check("в сообщениях нет истории разговора",
          [m["role"] for m in messages] == ["system", "system", "user"],
          json.dumps(messages, ensure_ascii=False)[:200])
    with_profile = support.messages("почему не работает вход?",
                                    profile="Профиль пользователя\nСтиль общения: без воды",
                                    ticket_block="ОБРАЩЕНИЕ TCK-1041")
    check("профиль клиента идёт отдельным блоком сразу после промпта",
          [m["role"] for m in with_profile] == ["system", "system", "system", "user"]
          and "без воды" in with_profile[1]["content"],
          json.dumps(with_profile, ensure_ascii=False)[:200])
    check("пустой профиль блока не создаёт",
          [m["role"] for m in support.messages("вопрос", profile="")] == ["system", "user"])
    check("данные обращения уходят отдельным системным блоком",
          "ДАННЫЕ ОБРАЩЕНИЯ (получены через MCP" in messages[1]["content"])
    check("выключенный сервер называется причиной, а не молчанием",
          "MCP" in support.hint() and "не подключён" in support.hint()
          and "потому что нет node" in support.hint("потому что нет node"))
    check("подсказка формата называет формы команды и где взять номер",
          support.COMMAND in support.format_hint()
          and "[номер обращения]" in support.format_hint()
          and "что у клиента в работе" in support.format_hint())
    # Зашитого примера номера в подсказках быть не должно: «TCK-1041» — это номер
    # из ДЕМО-данных, и в подсказке он читается как «правильный номер», которого у
    # человека нет (живой случай: на общий вопрос режим отвечал по документации и
    # приписывал «пришлите номер: /support TCK-1041 <вопрос>»).
    check("ни в подсказках, ни в правиле про номер нет зашитого примера",
          "TCK-1041" not in support.format_hint()
          and "TCK-1041" not in support.prompt_for(False, False)
          and "TCK-1041" not in support.prompt_for(True, True),
          support.format_hint())
    check("правило просьбы: описали проблему — предложи зафиксировать, но не заводи сам",
          "НЕ ПОПРОСИЛИ ЗАВЕСТИ ОБРАЩЕНИЕ" in support.prompt_for(True, True)
          and "заведи обращение по этому описанию" in support.prompt_for(True, True))
    check("предложение НЕ липнет к каждому ответу (списки и объяснения без него)",
          "ПРЕДЛОЖЕНИЕ ЭТО НЕ К КАЖДОМУ ОТВЕТУ" in support.prompt_for(True, True))
    check("правило про номер объясняет, где его взять, и не просит номер на общий вопрос",
          "5а) НОМЕР ОБРАЩЕНИЯ НЕ ВЫДУМЫВАЙ" in support.prompt_for(False, False)
          and "ЕСЛИ ВОПРОС НЕ ПРО ОБРАЩЕНИЕ" in support.prompt_for(False, False))


# ---------------------------------------------------------------------------
# [4] Отсечка: командные серверы не видны агенту
# ---------------------------------------------------------------------------
def check_gate():
    section("[4] Отсечка: поддержка работает только по команде")
    entry = mcp_store.find_server(mcp_store.SUPPORT) or {}
    check("сервер поддержки помечен как командный",
          mcp_store.is_command_only(entry)
          and mcp_store.chat_command_of(entry) == support.COMMAND, str(entry)[:200])
    check("у обычных серверов пометки нет",
          not mcp_store.is_command_only(mcp_store.find_server(mcp_store.GIT) or {}))
    check("агент не получает командные серверы",
          mcp_store.agent_enabled(["git", mcp_store.SUPPORT, "weather"])
          == ["git", "weather"],
          str(mcp_store.agent_enabled(["git", mcp_store.SUPPORT, "weather"])))
    check("включённый у проекта сервер поддержки агенту всё равно не виден",
          mcp_store.agent_enabled([mcp_store.SUPPORT]) == [])
    check("неизвестный реестру id НЕ отсекается (отсечка — про командность)",
          mcp_store.agent_enabled(["нет_такого", "git"]) == ["нет_такого", "git"],
          str(mcp_store.agent_enabled(["нет_такого", "git"])))


# ---------------------------------------------------------------------------
# [3] Заявитель — из профиля, а не от модели
# ---------------------------------------------------------------------------
def check_requester():
    section("[3] Заявитель обращения: подстановка из профиля (без сети и Node)")
    call_one = {"server": mcp_store.SUPPORT, "tool": chat.CREATE_TICKET_TOOL,
                "arguments": {}}
    chat._fill_requester(call_one, "Ирина Соколова")
    check("имя клиента подставлено в вызов создания обращения",
          call_one["arguments"].get(chat.REQUESTER_ARG) == "Ирина Соколова",
          json.dumps(call_one, ensure_ascii=False))
    for placeholder in ("", "клиент", "не указан", "  "):
        stub = {"server": mcp_store.SUPPORT, "tool": chat.CREATE_TICKET_TOOL,
                "arguments": {chat.REQUESTER_ARG: placeholder}}
        chat._fill_requester(stub, "Ирина Соколова")
        check("заглушка модели %r заменяется именем клиента" % placeholder,
              stub["arguments"][chat.REQUESTER_ARG] == "Ирина Соколова",
              json.dumps(stub, ensure_ascii=False))
    named = {"server": mcp_store.SUPPORT, "tool": chat.CREATE_TICKET_TOOL,
             "arguments": {chat.REQUESTER_ARG: "Дмитрий Ковалёв"}}
    chat._fill_requester(named, "Ирина Соколова")
    check("названный человеком заявитель НЕ перебивается",
          named["arguments"][chat.REQUESTER_ARG] == "Дмитрий Ковалёв",
          json.dumps(named, ensure_ascii=False))
    other_tool = {"server": mcp_store.SUPPORT, "tool": "search_tickets",
                  "arguments": {"query": "вход"}}
    chat._fill_requester(other_tool, "Ирина Соколова")
    check("чтения поддержки не трогаются",
          chat.REQUESTER_ARG not in other_tool["arguments"],
          json.dumps(other_tool, ensure_ascii=False))
    other_server = {"server": mcp_store.GIT, "tool": chat.CREATE_TICKET_TOOL,
                    "arguments": {}}
    chat._fill_requester(other_server, "Ирина Соколова")
    check("чужие серверы не трогаются", other_server["arguments"] == {},
          json.dumps(other_server, ensure_ascii=False))
    empty = {"server": mcp_store.SUPPORT, "tool": chat.CREATE_TICKET_TOOL,
             "arguments": {}}
    chat._fill_requester(empty, "")
    check("пустое имя в профиле не подставляется (сервер честно откажет)",
          empty["arguments"] == {}, json.dumps(empty, ensure_ascii=False))

    check("дополнитель зарегистрирован в клиенте MCP",
          chat._fill_requester in mcp_store.ARG_FILLERS,
          str(mcp_store.ARG_FILLERS))
    # Профиль читается тем же полем, что уходит в системный блок агента: имя
    # клиента в поддержке и «Имя» в профиле — одно и то же поле (иначе два
    # источника разошлись бы, и обращение завели бы «от кого-то другого»).
    profile = profile_store.create_profile(
        chat._profiles, {"profile_name": "Проверка", "user_name": "Ирина Соколова"})
    check("имя заявителя берётся из поля «Имя» текущего профиля",
          chat._requester_name() == "Ирина Соколова",
          "профиль: %s, имя: %r" % (profile.get("id"), chat._requester_name()))
    check("поле профиля — то же самое, что в списке сведений",
          profile_store.USER_KEY in profile_store.FIELD_KEYS)

    # Реестр дополнителей: сбой одного не отменяет остальные (иначе одна ошибка в
    # коде веб-слоя молча выключила бы подстановку во всех запросах). Ожидаемая
    # запись в журнал на время этой проверки глушится: трассировка сбоя в итоге
    # проверки читалась бы как настоящая ошибка проекта.
    def broken(_call):
        raise RuntimeError("проверка: дополнитель сломан")

    import logging

    quiet = logging.getLogger("app.ai.mcp")
    level = quiet.level
    quiet.setLevel(logging.CRITICAL)
    mcp_store.register_arg_filler(broken)
    try:
        survived = {"server": mcp_store.SUPPORT, "tool": chat.CREATE_TICKET_TOOL,
                    "arguments": {chat.REQUESTER_ARG: "клиент"}}
        chat._REQUESTER.set("Ирина Соколова")
        mcp_store.fill_arguments(survived)
        check("сбой дополнителя не мешает остальным и не роняет запрос",
              survived["arguments"][chat.REQUESTER_ARG] == "Ирина Соколова",
              json.dumps(survived, ensure_ascii=False))
        check("значение берётся и из contextvar запроса (путь без явного имени)",
              survived["arguments"][chat.REQUESTER_ARG] == "Ирина Соколова")
    finally:
        mcp_store.ARG_FILLERS.remove(broken)
        quiet.setLevel(level)
        chat._REQUESTER.set("")


# ---------------------------------------------------------------------------
# [5] Маршрут POST /api/agent/support (на заглушке модели, без сети)
# ---------------------------------------------------------------------------
LLM_CALLS = []
LLM_ANSWER = "Ответ поддержки (заглушка проверки)."


async def fake_call_llm_async(**kwargs):
    """Заглушка модели: запоминает вызов и отвечает заранее известным текстом."""
    LLM_CALLS.append(kwargs)
    return LLM_ANSWER, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def texts(events, kind):
    """Тексты событий одного вида (как в tools/check_dev_help.py)."""
    return [str(e.get("text") or "") for e in events if e.get("type") == kind]


def _system_texts(call):
    return "\n".join(str(m.get("content") or "") for m in (call.get("messages") or [])
                     if m.get("role") == "system")


def _tickets():
    """Файл обращений проверки (тот же, что видят инструменты сервера)."""
    with open(os.environ["TICKETS_FILE"], encoding="utf-8") as handle:
        return json.load(handle)


async def run_support(content, session_id=""):
    """Прогон POST /api/agent/support без сети: собирает события NDJSON-потока."""
    response = await chat.agent_support(ChatMessage(content=content,
                                                   session_id=session_id))
    events = []
    async for chunk in response.body_iterator:
        for line in str(chunk).splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


async def check_route():
    section("[5] Маршрут /api/agent/support: выбранное обращение, стоп до модели")
    llm_client.call_llm_async = fake_call_llm_async
    # Профиль-клиент: имя из него подставляет заявителя (см. раздел [3]).
    profile = chat._current_profile()
    if not profile:
        profile = profile_store.create_profile(chat._profiles,
                                               {"profile_name": "Клиент"})
    # Имя заявителя берётся из профиля: «Вячеслав» есть в образце данных, поэтому
    # созданное обращение привяжется к СУЩЕСТВУЮЩЕМУ клиенту u-1001. Стиль, формат
    # и ограничения клиента — в промпте поддержки (решение 15.10).
    profile = chat._current_profile() or {}
    profile_store.apply_fields(profile, {"user_name": "Вячеслав",
                                         "style": "как с мидл-специалистом",
                                         "limits": "без воды"})
    await chat.task_create(chat.TaskCreate(name="Поддержка (проверка)"))
    task = chat._current_task()
    session = workspace_store.create_session(task)
    sid = str(session["id"])

    # 1. Сервер поддержки выключен у проекта: причина названа, модель не звалась.
    LLM_CALLS.clear()
    events = await run_support("/support TCK-1041 почему не работает вход?", sid)
    start = [e for e in events if e.get("type") == "support_start"]
    check("сервер выключен: заявлено в событии старта",
          bool(start) and start[0]["support"] is False and start[0]["ticket"] == "TCK-1041",
          str(start))
    check("сервер выключен: причина названа с кнопкой «MCP»",
          "MCP" in " ".join(texts(events, "debug")) + " ".join(texts(events, "error")))
    check("сервер выключен: модель НЕ вызывалась",
          not LLM_CALLS and bool(texts(events, "error")), str(len(LLM_CALLS)))

    workspace_store.set_mcp_enabled(task, [mcp_store.SUPPORT])

    # 2. Несуществующее обращение: запрос ОСТАНАВЛИВАЕТСЯ до вызова модели.
    LLM_CALLS.clear()
    events = await run_support("/support TCK-9999 что там случилось?", sid)
    check("неизвестный номер: остановлено до модели", not LLM_CALLS, str(len(LLM_CALLS)))
    stop = " ".join(texts(events, "error"))
    check("неизвестный номер: в отказе назван номер и причина",
          "TCK-9999" in stop and "до вызова модели" in stop, stop[:300])

    # 3. Выбранное обращение: один вызов модели, данные обращения в промпте.
    LLM_CALLS.clear()
    events = await run_support("/support TCK-1041 почему не работает вход?", sid)
    system = _system_texts(LLM_CALLS[-1]) if LLM_CALLS else ""
    check("ответ по обращению пришёл", texts(events, "bot")
          and LLM_ANSWER in texts(events, "bot")[0], str(texts(events, "bot"))[:200])
    check("в промпте ИМЕННО выбранное обращение",
          "TCK-1041" in system and "Не работает вход по паролю" in system, system[:300])
    check("в промпте переписка и данные клиента",
          "кука" in system and "Клиент: Вячеслав" in system, system[:400])
    check("профиль клиента дошёл до модели (стиль и ограничения)",
          "без воды" in system and "как с мидл-специалистом" in system, system[:300])
    check("истории разговора в промпте НЕТ",
          "ИСТОРИЯ РАЗГОВОРА" not in system)
    check("диспетчера не спрашивали: вызов задал код (поиска в промпте нет)",
          "search_tickets" not in system)
    # ПРОЗРАЧНОСТЬ: человек должен ВИДЕТЬ, что ушло в модель и что записано, а не
    # угадывать по ответу (живой вопрос: «не могу понять, считывается ли контекст»).
    model_line = " ".join(text for text in texts(events, "debug")
                          if "В модель уходит" in text)
    check("в журнале видно, что и сколько ушло в модель",
          "данные обращения" in model_line and "профиль клиента" in model_line
          and "истории разговора и памяти задачи в промпте нет" in model_line,
          model_line[:300])
    check("базы знаний не подключены — сказано словами",
          any("не подключены" in text for text in texts(events, "debug")),
          " ".join(texts(events, "debug"))[:300])

    # 4. Вопрос без номера: набор вызовов выбирает диспетчер — но только по
    #    поддержке и только на чтение, если про изменения не просили.
    saved_decide = mcp_store.decide
    try:
        async def decide_writes(user_message, tools, call, **kwargs):
            return {"mode": "single", "done": False, "reason": "",
                    "calls": [{"server": support.SERVER_ID, "tool": support.CREATE_TOOL,
                               "arguments": {"subject": "Не приходит уведомление о новом обращении",
                                             "description": "Клиент пишет, что письмо о новом обращении не приходит."}}]}

        async def decide_reads(user_message, tools, call, **kwargs):
            return {"mode": "single", "done": False, "reason": "",
                    "calls": [{"server": support.SERVER_ID, "tool": support.SEARCH_TOOL,
                               "arguments": {"query": user_message[:120]}}]}

        async def decide_alien(user_message, tools, call, **kwargs):
            return {"mode": "single", "done": False, "reason": "",
                    "calls": [{"server": mcp_store.GIT, "tool": "git_status",
                               "arguments": {}}]}

        before = len(_tickets()["tickets"])
        mcp_store.decide = decide_writes
        LLM_CALLS.clear()
        events = await run_support(
            "/support заведи обращение: клиенту не приходит уведомление", sid)
        after = _tickets()
        created = [t for t in after["tickets"]
                   if t["subject"] == "Не приходит уведомление о новом обращении"]
        check("по прямой просьбе обращение создано",
              bool(created) and len(after["tickets"]) == before + 1,
              str([t["id"] for t in after["tickets"]][-2:]))
        check("заявитель подставлен из профиля клиента",
              bool(created) and created[0]["user_id"] == "u-1001",
              json.dumps(created[:1], ensure_ascii=False)[:200])
        write_line = " ".join(text for text in texts(events, "debug")
                              if "Изменение" in text)
        check("в журнале видно, что обращение заведено и с каким номером",
              "create_ticket" in write_line and "TCK-" in write_line
              and "запись на диске" in write_line, write_line[:300])
        check("подтверждение инструмента ушло модели",
              "ОБРАЩЕНИЕ СОЗДАНО" in _system_texts(LLM_CALLS[-1]),
              _system_texts(LLM_CALLS[-1])[-300:])

        mcp_store.decide = decide_writes
        LLM_CALLS.clear()
        events = await run_support("/support что у клиента в работе?", sid)
        check("без просьбы изменение не выполняется, а называется причиной",
              any("изменение" in text for text in texts(events, "debug"))
              and len(_tickets()["tickets"]) == before + 1,
              " ".join(texts(events, "debug"))[:300])

        mcp_store.decide = decide_reads
        LLM_CALLS.clear()
        events = await run_support("/support что у клиента в работе?", sid)
        check("без номера чтение поддержки идёт и ответ приходит",
              any("search_tickets" in text for text in texts(events, "debug"))
              and bool(texts(events, "bot")),
              " ".join(texts(events, "debug"))[:300])

        mcp_store.decide = decide_alien
        LLM_CALLS.clear()
        events = await run_support("/support что там с репозиторием?", sid)
        check("чужой сервер в режиме поддержки отброшен",
              any("только сервер" in text for text in texts(events, "debug")),
              " ".join(texts(events, "debug"))[:300])
    finally:
        mcp_store.decide = saved_decide

    # 5. Не команда — отказ 400 (иначе обычный текст уходил бы в режим поддержки).
    try:
        await chat.agent_support(ChatMessage(content="обычный вопрос", session_id=sid))
        check("не команда — отказ", False, "исключения не было")
    except Exception as exc:                        # noqa: BLE001 — ждём HTTPException
        check("не команда — отказ 400", getattr(exc, "status_code", None) == 400,
              str(exc)[:200])


def main():
    global SANDBOX
    parser = argparse.ArgumentParser(
        description="Проверка MCP-сервера «Поддержка» и правила /help")
    parser.add_argument("--keep", action="store_true",
                        help="не удалять временный каталог с файлом данных")
    options = parser.parse_args()

    SANDBOX = tempfile.mkdtemp(prefix="check_support_mcp_")
    # Временный файл данных задаётся ДО первого вызова: сервер запускается
    # отдельным процессом и наследует окружение.
    os.environ["TICKETS_FILE"] = os.path.join(SANDBOX, "tickets.json")
    print("Живая проверка MCP-сервера «Поддержка» (клиенты и обращения)")
    print(f"  --   временный файл данных: {os.environ['TICKETS_FILE']}")
    try:
        check_server()
        check_command_rules()
        check_gate()
        check_requester()
        asyncio.run(check_route())
    finally:
        entry = mcp_store.find_server(mcp_store.SUPPORT) or {}
        print(f"\n  --   запись реестра: id={entry.get('id')}, name={entry.get('name')}, "
              f"script={entry.get('script')}, source={entry.get('source')}")
        if not options.keep:
            shutil.rmtree(SANDBOX, ignore_errors=True)
            shutil.rmtree(_TMP, ignore_errors=True)

    print("\nИтог: " + (f"ПРОВАЛЕНО проверок: {len(FAILED)} → {FAILED}"
                       if FAILED else "все проверки пройдены"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
