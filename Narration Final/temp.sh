cd /mnt/hard/home/toroghi/NarrationStack
bash containers/build_existing.sh
docker compose -f compose.existing.yaml up -d --no-build --pull never
