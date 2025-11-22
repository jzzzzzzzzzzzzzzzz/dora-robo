conda activate op

python operating_platform/core/main.py \
    --robot.type=so101 \
    --record.repo_id="so101-1108" \
    --record.single_task="so101 arm grab ori"

