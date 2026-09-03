"""Spans outside traces reach metering processors only."""

from typing import Any

from onyx.tracing.framework.processor_interface import TracingProcessor
from onyx.tracing.framework.provider import DefaultTraceProvider
from onyx.tracing.framework.span_data import GenerationSpanData
from onyx.tracing.framework.spans import Span
from onyx.tracing.framework.traces import Trace


class _CaptureProcessor(TracingProcessor):
    def __init__(self, process_without_trace: bool) -> None:
        self.process_without_trace = process_without_trace
        self.ended_spans: list[Span[Any]] = []

    def on_trace_start(self, trace: Trace) -> None:
        pass

    def on_trace_end(self, trace: Trace) -> None:
        pass

    def on_span_start(self, span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        self.ended_spans.append(span)

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


def test_orphan_generation_reaches_only_opted_in_processors() -> None:
    provider = DefaultTraceProvider()
    metering_processor = _CaptureProcessor(process_without_trace=True)
    external_processor = _CaptureProcessor(process_without_trace=False)
    provider.register_processor(metering_processor)
    provider.register_processor(external_processor)

    span = provider.create_span(GenerationSpanData(model="claude-sonnet"))
    with span:
        span.span_data.usage = {"input_tokens": 5, "output_tokens": 2}

    assert metering_processor.ended_spans == [span]
    assert external_processor.ended_spans == []
