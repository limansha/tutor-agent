# Render Database Backup

These local-only scripts back up and restore the Render PostgreSQL database.
They are intentionally ignored by git along with the dump directory.

## Dump location

Backups are always written to:

```text
<repository-root>/render_db_dumps/
```

The restore script only accepts dump files from that directory.

## Create a backup

Use the Render dashboard's **external** database connection URL:

```bash
python scripts/render_db/backup.py 'postgresql://...'
```

The script creates a timestamped `.dump` file and verifies it with
`pg_restore --list`.

## Restore into a new Render database

1. Create a new Render Postgres database manually.
2. Copy its external connection URL.
3. Restore the selected dump:

```bash
TARGET_DATABASE_URL='NEW_RENDER_EXTERNAL_DATABASE_URL' \
  python scripts/render_db/restore.py \
  tutor-agent-YYYYMMDD-HHMMSS.dump --confirm
```

Run this from the repository root. Pass only the dump filename; the restore
script finds it in `render_db_dumps/`. The target must be the new empty
database. The restore does not use `--clean` and does not drop existing tables.

4. Update the Render web service's `DATABASE_URL` to the new database.
5. Check `/api/health`, `/api/questions`, and `/api/check-answer`.

The PostgreSQL client tools `pg_dump` and `pg_restore` must be installed
locally. Database URLs are supplied through environment variables so they are
not committed to the repository.
