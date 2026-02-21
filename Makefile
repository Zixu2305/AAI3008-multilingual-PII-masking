build:
	docker compose --env-file .env -f docker/docker-compose.yml build

shell:
	docker compose --env-file .env -f docker/docker-compose.yml run --rm app bash

smoke:
	docker compose --env-file .env -f docker/docker-compose.yml run --rm app python scripts/00_smoke_asr.py

run_asr:
	docker compose --env-file .env -f docker/docker-compose.yml run --rm app python scripts/10_run_asr.py --config configs/asr.yaml

run_pii:
	docker compose --env-file .env -f docker/docker-compose.yml run --rm app python scripts/20_run_pii.py --config configs/pii.yaml

mask:
	docker compose --env-file .env -f docker/docker-compose.yml run --rm app python scripts/30_mask_audio.py --config configs/pii.yaml

eval:
	docker compose --env-file .env -f docker/docker-compose.yml run --rm app python scripts/40_eval.py --config configs/eval.yaml
