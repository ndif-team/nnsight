from . import Backend


class LocalMixin:

    def local_backend_execute(self):

        raise NotImplementedError()


class LocalBackend(Backend):

    def __call__(self, obj: LocalMixin):

        obj.local_backend_execute()
