# Post-Proc Custom Aggregation

## Overview

Adds custom aggregation task to existing postproc app. Runs on Jasmin after postproc completes.

## Integration Steps

### 1. Merge Into Your Workflow

Merge these files into your existing `app/postproc/`:
- `bin/aggregate.sh` - Bash wrapper
- `file/postproc/aggregate_config.json` - Your aggregation config
- `postproc.rc` - Cylc config (include in flow.cylc)

Then edit `rose-app.conf` as described below.

### 2. Add to rose-app.conf

**Merge these sections into your existing `app/postproc/rose-app.conf`:**

```ini
[env]
TRANSFER_DIR=/your/jasmin/archive/path
MODEL_NAME=${CYLC_WORKFLOW_NAME}
AGGREGATE_OUTPUT_FILENAME=aggregated.nc
OUTPUT_FILE=${TRANSFER_DIR}/${MODEL_NAME}/${AGGREGATE_OUTPUT_FILENAME}
AGGREGATE_CONFIG=$ROSE_DATA/postproc/aggregate_config.json

[command]
aggregate=bin/aggregate.sh

[file:postproc/aggregate_config.json]
source=file:postproc/aggregate_config.json
```

**Modify your existing `[namelist:pptransfer]`:**
```ini
# FROM:
[namelist:pptransfer]
transfer_dir=/your/old/path

# TO:
[namelist:pptransfer]
transfer_dir=${TRANSFER_DIR}
```

### 3. Edit bin/aggregate.sh

Modify line 5 to setup your Python environment on Jasmin:
```bash
. /path/to/your/venv/bin/activate
```

### 4. Add to flow.cylc

```cylc
{% include 'app/postproc/postproc.rc' %}
```

### 5. Create Your Python Script

```bash
cp scripts/aggregate.py.example $OPTCLIMTOP/scripts/aggregate.py
# Edit for your processing logic
```

### 6. Test

```bash
cylc validate /path/to/workflow
```

## Files

| File | Purpose |
|------|---------|
| `rose-app.conf` | Merge [env], [command], [file] sections into existing config |
| `postproc.rc` | Cylc config for aggregation task (include in flow.cylc) |
| `bin/aggregate.sh` | Bash wrapper (edit Python env setup) |
| `file/postproc/aggregate_config.json` | Your aggregation config |
| `scripts/aggregate.py.example` | Python template |

## See Also

- Main documentation: `../um_rose_post_process.md`
