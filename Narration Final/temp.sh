cd /mnt/hard/home/toroghi/NarrationStack
bash containers/move_and_build_existing.sh
bash containers/verify_existing.sh
docker compose --env-file existing-paths.env \
  -f compose.existing.yaml up -d --no-build --pull never
