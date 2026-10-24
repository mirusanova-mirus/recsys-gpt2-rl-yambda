PYTHON ?= python3
CONFIG ?= configs/main.json

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
	$(PYTHON) -u -m seqrl.examples --config $(CONFIG)
audit: rollout tests
	$(PYTHON) -u -m seqrl.validate --config $(CONFIG) --final
tests:
	$(PYTHON) -m unittest discover -s tests -v
report: audit
	$(PYTHON) -m seqrl.report --config $(CONFIG)
	$(PYTHON) -m seqrl.pdf_report
package: report
	$(PYTHON) -m seqrl.package --config $(CONFIG)

.PHONY: latex
latex:
	$(MAKE) -C report all
