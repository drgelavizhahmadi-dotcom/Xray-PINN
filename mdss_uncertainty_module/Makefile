.PHONY: setup run-api generate-demo test lint format

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e .
	mkdir -p data/results logs

run-api:
	.venv/bin/uvicorn mdss_uncertainty.api:app --reload --host 0.0.0.0 --port 8000

generate-demo:
	.venv/bin/python -c "from mdss_uncertainty.report_generator import ReportGenerator; \
		g = ReportGenerator(); \
		g.save_demo_set( \
			{'uncertainty': 0.05, 'confidence': 0.92, 'ci_lower': 0.88, 'ci_upper': 0.96, 'risk_level': 'LOW', 'oversight_action': 'Proceed', 'article_14_status': 'No notification'}, \
			{'uncertainty': 0.15, 'confidence': 0.78, 'ci_lower': 0.72, 'ci_upper': 0.84, 'risk_level': 'MEDIUM', 'oversight_action': 'Encourage double reading', 'article_14_status': 'Radiologist notification recommended'}, \
			{'uncertainty': 0.35, 'confidence': 0.55, 'ci_lower': 0.42, 'ci_upper': 0.68, 'risk_level': 'HIGH', 'oversight_action': 'Mandatory review', 'article_14_status': 'Article 14 oversight required'} \
		)"

test:
	.venv/bin/pytest tests/ -v

lint:
	.venv/bin/ruff check src/ && .venv/bin/black --check src/

format:
	.venv/bin/ruff check --fix src/ && .venv/bin/black src/
