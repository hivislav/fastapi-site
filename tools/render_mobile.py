"""Рендер страницы чата в системном WebKit на телефонной ширине.

Зачем: мобильную вёрстку нельзя проверить ни правилами CSS (jsdom не считает
раскладку), ни на глаз по коду — нужен настоящий движок, который скажет, что
именно вылезает за экран и сколько высоты остаётся окну чата.

Скрипт открывает страницу в WKWebView нужного размера, выполняет в ней замер
(какие элементы шире экрана, сколько высоты у шапки, полосы задачи, переписки и
поля ввода), печатает отчёт и сохраняет снимок экрана.

Запуск:
    PYTHONPATH=/tmp/shotlib ./venv/bin/python tools/render_mobile.py \
        --url https://127.0.0.1:8000/ --width 390 --height 844 --out /tmp/m1.png
"""

import argparse
import json
import sys
import time

import objc
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSBitmapImageFileTypePNG,
    NSBitmapImageRep,
    NSMakeRect,
    NSRunLoop,
    NSDate,
    NSDefaultRunLoopMode,
    NSWindow,
    NSWindowStyleMaskBorderless,
)
from Foundation import NSObject, NSURLCredential, NSURLRequest, NSURL
from WebKit import (
    WKWebView,
    WKWebViewConfiguration,
    WKSnapshotConfiguration,
)

# Замер в самой странице: что вылезает за экран и сколько высоты у ключевых
# блоков. Возвращается строкой JSON — так результат не зависит от типов моста.
AUDIT_JS = r"""
(() => {
  const vw = window.innerWidth, vh = window.innerHeight;
  const name = (el) => {
    let s = el.tagName.toLowerCase();
    if (el.id) s += '#' + el.id;
    const cls = (typeof el.className === 'string' ? el.className : '').trim();
    if (cls) s += '.' + cls.split(/\s+/).slice(0, 3).join('.');
    return s;
  };
  const rect = (el) => {
    const r = el.getBoundingClientRect();
    return { h: Math.round(r.height), w: Math.round(r.width),
             left: Math.round(r.left), top: Math.round(r.top) };
  };
  const shown = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return false;
    const st = getComputedStyle(el);
    if (st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') return false;
    return true;
  };
  // Дети блока: кто сколько высоты съедает.
  const kids = (sel) => {
    const root = document.querySelector(sel);
    if (!root) return [];
    return Array.from(root.children).filter(shown).map((el) => {
      const r = rect(el);
      return { el: name(el), h: r.h, w: r.w, top: r.top,
               text: (el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 24) };
    });
  };
  // Что реально вылезает за экран: ящики, сдвинутые за край, в счёт не идут —
  // они там и должны быть, пока закрыты.
  const overflow = [];
  document.querySelectorAll('body *').forEach((el) => {
    if (!shown(el)) return;
    const r = el.getBoundingClientRect();
    if (r.left >= vw || r.right <= 0) return;          // за краем — это ящик
    if (r.right > vw + 1) {
      overflow.push({ el: name(el), left: Math.round(r.left), right: Math.round(r.right),
                      w: Math.round(r.width),
                      text: (el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 30) });
    }
  });
  const sidebar = document.querySelector('.sidebar');
  return JSON.stringify({
    vw, vh,
    bodyClass: document.body.className,
    debug: { fired: window.__fired || 0, cls: window.__cls || '', err: window.__err || '', log: window.__log || [] },
    // Что окажется под пальцем в центре экрана: так проверяется, что модалка
    // выше выдвижной панели, а не спрятана под ней.
    hitTest: (() => {
      const el = document.elementFromPoint(Math.round(vw / 2), Math.round(vh / 2));
      if (!el) return null;
      const overlay = el.closest ? el.closest('.modal-overlay') : null;
      return { tag: el.tagName.toLowerCase(), cls: (el.className || '').toString().slice(0, 40),
               inModal: overlay ? overlay.id : null };
    })(),
    overlays: (() => {
      return Array.from(document.querySelectorAll('.modal-overlay')).map((overlay) => {
        const box = overlay.querySelector('.modal-box');
        const r = overlay.getBoundingClientRect();
        const rb = box ? box.getBoundingClientRect() : null;
        const st = getComputedStyle(overlay);
        return {
          id: overlay.id, hidden: overlay.hidden, position: st.position, zIndex: st.zIndex,
          overlay: { left: Math.round(r.left), top: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height) },
          box: rb ? { left: Math.round(rb.left), top: Math.round(rb.top), w: Math.round(rb.width), h: Math.round(rb.height) } : null,
          parent: overlay.parentElement ? (overlay.parentElement.className || overlay.parentElement.tagName) : null,
          chatBackdropFilter: getComputedStyle(document.querySelector('.chat')).backdropFilter,
        };
      }).filter((m) => !m.hidden);
    })(),
    messagesScrollTop: (() => {
      const m = document.querySelector('.messages');
      return { top: Math.round(m.scrollTop), scrollHeight: Math.round(m.scrollHeight),
               clientHeight: Math.round(m.clientHeight) };
    })(),
    probe: (() => {
      const header = document.querySelector('.chat-header');
      const sidebar = document.querySelector('.sidebar');
      const h = getComputedStyle(header), s = getComputedStyle(sidebar);
      const rules = [];
      for (const sheet of document.styleSheets) {
        let list; try { list = sheet.cssRules; } catch (e) { continue; }
        for (const rule of list) {
          if (!rule.cssRules) continue;
          const text = (rule.media && rule.media.mediaText) || rule.conditionText || '';
          if (!/1024px/.test(text)) continue;
          for (const inner of rule.cssRules) rules.push(inner.selectorText || '');
        }
      }
      return {
        media1024: window.matchMedia('(max-width: 1024px)').matches,
        headerMaxHeight: h.maxHeight, headerHeight: h.height, headerOpacity: h.opacity,
        sidebarTransform: s.transform,
        hasHiddenRule: rules.indexOf('body.header-hidden .chat-header') !== -1,
        hasDrawerRule: rules.indexOf('body.drawer-open .sidebar') !== -1,
        mobileRuleCount: rules.length,
        matchHidden: !!document.querySelector('body.header-hidden .chat-header'),
        matchDrawer: !!document.querySelector('body.drawer-open .sidebar'),
        ruleText: (() => {
          const out = [];
          for (const sheet of document.styleSheets) {
            let list; try { list = sheet.cssRules; } catch (e) { continue; }
            for (const rule of list) {
              if (!rule.cssRules) continue;
              for (const inner of rule.cssRules) {
                const st = inner.selectorText || '';
                if (st === 'body.header-hidden .chat-header' || st === 'body.drawer-open .sidebar') {
                  out.push({ sel: st, len: inner.style ? inner.style.length : -1,
                             text: inner.cssText.slice(0, 150) });
                }
              }
            }
          }
          return out;
        })(),
      };
    })(),
    historyLength: history.length,
    sidebarLeft: sidebar ? Math.round(sidebar.getBoundingClientRect().left) : null,
    expertVisible: (() => {
      const el = document.getElementById('expert-toggle');
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return r.width > 1 && r.height > 1 && getComputedStyle(el).display !== 'none';
    })(),
    docScrollWidth: document.documentElement.scrollWidth,
    header: rect(document.querySelector('.chat-header')),
    headerKids: kids('.chat-header'),
    headerTopKids: kids('.chat-header-top'),
    settingsKids: kids('.chat-settings'),
    taskMachineKids: kids('.task-machine'),
    tmRows: (() => {
      const r = (sel) => {
        const el = document.querySelector(sel);
        if (!el) return null;
        const b = el.getBoundingClientRect();
        const st = getComputedStyle(el);
        return { top: Math.round(b.top), bottom: Math.round(b.bottom), h: Math.round(b.height),
                 w: Math.round(b.width), display: st.display };
      };
      const machine = document.querySelector('.task-machine');
      return { track: r('.tm-track'), actions: r('.tm-actions'), meta: r('.tm-meta'),
               shadow: machine ? getComputedStyle(machine).boxShadow.slice(0, 30) : null };
    })(),
    inputRowKids: kids('.input-row'),
    messages: rect(document.querySelector('.messages')),
    heights: {
      taskMachine: rect(document.querySelector('.task-machine')).h,
      chat: rect(document.querySelector('.chat')).h,
    },
    overflow: overflow.slice(0, 20),
    overflowTotal: overflow.length,
  });
})()
"""


class Delegate(NSObject):
    """Ждём загрузку и принимаем самоподписанный сертификат (он у нас свой)."""

    def init(self):
        self = objc.super(Delegate, self).init()
        self.done = False
        self.error = None
        return self

    def webView_didFinishNavigation_(self, webView, navigation):
        self.done = True

    def webView_didFailNavigation_withError_(self, webView, navigation, error):
        self.error = str(error)
        self.done = True

    def webView_didFailProvisionalNavigation_withError_(self, webView, navigation, error):
        self.error = str(error)
        self.done = True

    def webView_didReceiveAuthenticationChallenge_completionHandler_(
            self, webView, challenge, handler):
        trust = challenge.protectionSpace().serverTrust()
        handler(1, NSURLCredential.credentialForTrust_(trust))


def pump(seconds):
    """Крутим цикл событий: без него WebKit не грузит и не рисует."""
    end = time.time() + seconds
    while time.time() < end:
        NSRunLoop.currentRunLoop().runMode_beforeDate_(
            NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.05))


def main():
    parser = argparse.ArgumentParser(description="Рендер страницы в WebKit на телефонной ширине")
    parser.add_argument("--url", default="https://127.0.0.1:8000/")
    parser.add_argument("--width", type=int, default=390)
    parser.add_argument("--height", type=int, default=844)
    parser.add_argument("--out", default="/tmp/shot.png")
    parser.add_argument("--js", default="", help="код, выполняемый до снимка (например, открыть ящик)")
    parser.add_argument("--wait", type=float, default=6.0, help="сколько ждать после загрузки")
    args = parser.parse_args()

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    frame = NSMakeRect(0, 0, args.width, args.height)
    web = WKWebView.alloc().initWithFrame_configuration_(frame, WKWebViewConfiguration.alloc().init())
    delegate = Delegate.alloc().init()
    web.setNavigationDelegate_(delegate)

    window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
    window.setContentView_(web)
    window.orderFrontRegardless()

    web.loadRequest_(NSURLRequest.requestWithURL_(NSURL.URLWithString_(args.url)))
    deadline = time.time() + 30
    while not delegate.done and time.time() < deadline:
        pump(0.1)
    if delegate.error:
        print("ОШИБКА загрузки:", delegate.error)
    pump(args.wait)

    # Переходы выключаются ПЕРЕД сценарием: окно WebKit здесь невидимое, и
    # анимации в нём не проигрываются — замер тогда видит состояние «до
    # перехода» (ящик так и оставался за краем, хотя класс уже стоял). На сам
    # CSS это не влияет: переходы — только плавность.
    web.evaluateJavaScript_completionHandler_(
        "(() => { const st = document.createElement('style');"
        " st.textContent = '*{transition:none !important}';"
        " document.head.appendChild(st); })()", lambda value, err: None)
    pump(0.4)

    if args.js:
        result = {}
        web.evaluateJavaScript_completionHandler_(args.js, lambda value, err: result.update(v=value, e=err))
        pump(1.5)
        if result.get("e"):
            print("ОШИБКА js:", result["e"])

    audit = {}
    web.evaluateJavaScript_completionHandler_(AUDIT_JS, lambda value, err: audit.update(v=value, e=err))
    pump(1.5)
    if audit.get("v"):
        print(json.dumps(json.loads(audit["v"]), ensure_ascii=False, indent=1))
    elif audit.get("e"):
        print("ОШИБКА замера:", audit["e"])

    shot = {}
    config = WKSnapshotConfiguration.alloc().init()
    config.setRect_(web.bounds())

    def on_snapshot(image, error):
        if error or image is None:
            shot["error"] = str(error)
            return
        rep = NSBitmapImageRep.imageRepWithData_(image.TIFFRepresentation())
        data = rep.representationUsingType_properties_(NSBitmapImageFileTypePNG, {})
        shot["ok"] = bool(data.writeToFile_atomically_(args.out, True))

    web.takeSnapshotWithConfiguration_completionHandler_(config, on_snapshot)
    pump(4)
    print("снимок:", args.out, "записан" if shot.get("ok") else "НЕ записан", shot.get("error", ""))
    return 0 if shot.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
