"""Shared exceptions. `.message` is safe to show in the UI; `.http_status` maps to the HTTP response."""


class PipelineError(Exception):
    http_status = 500

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class UnsupportedFileError(PipelineError):
    http_status = 415


class EmptyFileError(PipelineError):
    http_status = 400


class UnreadableAudioError(PipelineError):
    http_status = 422


class NoSpeechError(PipelineError):
    http_status = 422


class LLMStageError(PipelineError):
    http_status = 502
