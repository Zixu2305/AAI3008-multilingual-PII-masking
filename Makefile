build:
	docker compose --env-file .env -f docker/docker-compose.yml build

shell:
	docker compose --env-file .env -f docker/docker-compose.yml run --rm app bash

smoke:
	docker compose --env-file .env -f docker/docker-compose.yml run --rm app python -m src.cli asr --config configs/asr.yaml

run_asr:
	docker compose --env-file .env -f docker/docker-compose.yml run --rm app python -m src.cli asr --config configs/asr.yaml

run_asr_eval:
	docker compose --env-file .env -f docker/docker-compose.yml run --rm app python -m src.cli asr-eval --config configs/asr_eval.yaml

run_pii:
	docker compose --env-file .env -f docker/docker-compose.yml run --rm app python -m src.cli pii --config configs/pii.yaml

run_pii_eval:
	docker compose --env-file .env -f docker/docker-compose.yml run --rm app python -m src.cli pii-eval --config configs/pii_eval_wikiann.yaml

run_pii_eval_gold:
	docker compose --env-file .env -f docker/docker-compose.yml run --rm app python -m src.cli pii-eval-gold --config configs/pii_eval_gold.yaml

mask:
	docker compose --env-file .env -f docker/docker-compose.yml run --rm app python -m src.cli mask --config configs/pii.yaml

eval:
	docker compose --env-file .env -f docker/docker-compose.yml run --rm app python -m src.cli eval --config configs/eval.yaml
