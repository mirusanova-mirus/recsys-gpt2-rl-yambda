PYTHON ?= python3
CONFIG ?= configs/main.json
.NOTPARALLEL:

.PHONY: reproduce data build sft reward rl evaluate rollout audit tests report package

# Invoke sequentially even if make is called with -j: later stages depend on fits.
reproduce: package
data:
	$(PYTHON) -u -m seqrl.data
build: data
	$(PYTHON) -u -m seqrl.build --config $(CONFIG)
sft: build
	$(PYTHON) -u -m seqrl.sft --config $(CONFIG)
reward: sft
	$(PYTHON) -u -m seqrl.reward --config $(CONFIG)
rl: reward
	$(PYTHON) -u -m seqrl.rl --config $(CONFIG)
evaluate: rl
	$(PYTHON) -u -m seqrl.evaluate --config $(CONFIG)
rollout: evaluate
	$(PYTHON) -u -m seqrl.rollout --config $(CONFIG)
	$(PYTHON) -u -m seqrl.session_evaluate --config $(CONFIG)
	$(PYTHON) tools/run_generated.py --output EXAMPLES.md -- $(PYTHON) -u -m seqrl.examples --config $(CONFIG)
audit: rollout tests
	$(PYTHON) -u -m seqrl.validate --config $(CONFIG) --final
tests:
	$(PYTHON) -m unittest discover -s tests -v
report: audit
	$(MAKE) generated-report PYTHON=$(PYTHON) CONFIG=$(CONFIG)
generated-report:
	$(PYTHON) tools/run_generated.py --output REPORT.md -- $(PYTHON) -m seqrl.report --config $(CONFIG)
check:
	$(PYTHON) tools/check_submission.py
package: report check
	$(PYTHON) tools/package_submission.py --output submission/recsys_rl.zip
bundle: check
	$(PYTHON) tools/package_submission.py --output submission/recsys_rl.zip

.PHONY: check bundle generated-report

.PHONY: latex
latex:
	$(MAKE) -C report all
