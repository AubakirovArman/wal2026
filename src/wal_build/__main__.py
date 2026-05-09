from . import *

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m wal_build <command> [args...]")
        print("")
        print("Commands:")
        print("  init <base_model>          Initialize WAL project")
        print("  edit add <facts.json>      Add edit recipe")
        print("  status                     Show project status")
        print("  tag <name> [build_id]      Tag a build")
        print("  rollback <tag>             Rollback to tag")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "init" and len(sys.argv) >= 3:
        init_project(sys.argv[2])
    elif cmd == "edit" and sys.argv[2] == "add" and len(sys.argv) >= 4:
        strategy = sys.argv[4] if len(sys.argv) >= 5 else "auto"
        add_edit(sys.argv[3], strategy)
    elif cmd == "status":
        status()
    elif cmd == "tag" and len(sys.argv) >= 3:
        build_id = sys.argv[3] if len(sys.argv) >= 4 else None
        tag_version(sys.argv[2], build_id)
    elif cmd == "rollback" and len(sys.argv) >= 3:
        rollback(sys.argv[2])
    else:
        print(f"❌ Unknown command: {' '.join(sys.argv[1:])}")
