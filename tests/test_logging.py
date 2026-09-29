"""racerts never prints; verbose and -v/-vv only show more, never less."""

import logging
import threading
import time

import racerts
from racerts.cli import main
from racerts.utils.log import show_messages, verbose_logging

from .conftest import EX

TS = ["-atoms", "3", "4", "5", "-smiles", "CCCCCC=C", "-n", "3"]


def test_verbose_does_not_hide_debug_output(caplog):
    with show_messages(logging.DEBUG):
        racerts.ConformerGenerator(verbose=True).generate_conformers(
            EX, 0, [3, 4, 5], input_smiles=["CCCCCC=C"], number_of_conformers=3
        )

    assert any(record.levelno == logging.DEBUG for record in caplog.records)


def test_legacy_cli_shows_debug_output_with_vv(caplog, tmp_path):
    main([EX, *TS, "-v", "-v", "-o", str(tmp_path / "out.xyz")])

    assert any(record.levelno == logging.DEBUG for record in caplog.records)


def test_verbose_calls_in_threads_restore_the_logger():
    logger = logging.getLogger("racerts")
    level, handlers = logger.level, list(logger.handlers)

    def work(pause):
        with verbose_logging(True):
            time.sleep(pause)

    threads = [threading.Thread(target=work, args=(0.01 * i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert logger.level == level and logger.handlers == handlers
