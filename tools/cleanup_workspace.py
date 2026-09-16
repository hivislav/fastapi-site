"""Разовая чистка файла workspace: убираем данные профилей, которых больше нет.

Запуск (показывает, что будет сделано, и НЕ меняет файл):
    ./venv/bin/python tools/cleanup_workspace.py
Реально применить:
    ./venv/bin/python tools/cleanup_workspace.py --apply
Применить без копии файла (проект в активной разработке, данные не жалко):
    ./venv/bin/python tools/cleanup_workspace.py --apply --no-backup

Что делает:
  * удаляет задачи, чей владелец (profile) отсутствует в data/profiles.json —
    вместе с их диалогами (раньше удаление профиля оставляло их в файле);
  * чистит указатели active_tasks на несуществующие задачи и мёртвый
    active_task прежних версий;
  * чистит память профилей, которых больше нет;
  * чистит записи задач без имени (их отбрасывает нормализация).
Данные существующих профилей не трогаются.
"""

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config  # noqa: E402
from app.ai import profiles as profile_store  # noqa: E402
from app.ai import workspace as workspace_store  # noqa: E402


def main() -> int:
    apply_changes = "--apply" in sys.argv
    store = profile_store.load_profiles()
    known = {profile["id"] for profile in store.get("profiles", [])}
    raw = workspace_store.load_workspace()
    tasks = raw.get("tasks", [])

    orphans = [t for t in tasks if workspace_store.task_owner(t) not in known]
    kept = [t for t in tasks if workspace_store.task_owner(t) in known]
    dead_active = {k: v for k, v in (raw.get("active_tasks") or {}).items()
                   if k not in known or v not in [t["id"] for t in kept]}
    dead_memory = [k for k in (raw.get("long_term_by_profile") or {}) if k not in known]

    print(f"файл workspace: {config.AGENT_WORKSPACE_FILE}")
    print(f"профилей сейчас: {len(known)} — {sorted(known)}")
    print(f"задач в файле: {len(tasks)}; из них без своего профиля: {len(orphans)}")
    for task in orphans:
        sessions = task.get("sessions", [])
        replies = sum(len(s.get("dialog", {}).get("messages", [])) for s in sessions)
        print(f"  - {task['id']} «{task['name']}» (профиль {task.get('profile') or '—'}): "
              f"диалогов {len(sessions)}, реплик {replies}")
    print(f"мёртвых указателей active_tasks: {len(dead_active)} {dead_active}")
    print(f"память удалённых профилей: {dead_memory}")

    if not (orphans or dead_active or dead_memory):
        print("чистить нечего — файл в порядке.")
        return 0

    if not apply_changes:
        print("\nэто предпросмотр. Чтобы применить: --apply")
        return 0

    backup = None
    if "--no-backup" not in sys.argv:
        backup = config.AGENT_WORKSPACE_FILE + ".bak"
        shutil.copy(config.AGENT_WORKSPACE_FILE, backup)
    raw["tasks"] = kept
    for key in dead_active:
        raw.get("active_tasks", {}).pop(key, None)
    for key in dead_memory:
        raw.get("long_term_by_profile", {}).pop(key, None)
    for key in list((raw.get("active_tasks") or {})):
        if key not in known:
            raw["active_tasks"].pop(key, None)
    workspace_store.save_workspace(raw)
    after = workspace_store.load_workspace()
    print(f"\nготово: задач осталось {len(after.get('tasks', []))}, "
          f"указатели {after.get('active_tasks')}; "
          + (f"бэкап: {backup}" if backup else "бэкап не создавался (--no-backup)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
