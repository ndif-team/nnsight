"""Module calls and replacements shared across the workers of one forward.

With a `Coalescer` on the interleaver, a module a block calls ad hoc runs once
for every worker making the same call at the same moment, and each worker gets
its own rows of the result (``intervention/coalesce.py``), when the runtime's
predicate says the call is token-wise or the module was registered. Replacements the
workers assign at one location are written into the value in one pass
(``Batcher.gather_swap`` / ``assemble_swap``).

The toy models stack their invokes on dim 0 and use module names no hosted
model has, in a tensor-output and a tuple-output variant.
"""

from __future__ import annotations

import pytest
import torch

import nnsight
from nnsight import NNsight
from nnsight.intervention.coalesce import Coalescer, Unsplittable, split

WIDTH = 8


class Block(torch.nn.Module):
    def __init__(self, tuple_output: bool) -> None:
        super().__init__()
        self.mix = torch.nn.Linear(WIDTH, WIDTH)
        self.tuple_output = tuple_output

    def forward(self, hidden):
        hidden = torch.tanh(self.mix(hidden))
        return (hidden, None) if self.tuple_output else hidden


class RowMean(torch.nn.Module):
    """Mixes rows: each output row is its input row minus the mean over rows."""

    def forward(self, hidden):
        return hidden - hidden.mean(dim=0, keepdim=True)


class RowPosition(torch.nn.Module):
    """Row-wise, but a row's output depends on where the row sits."""

    def forward(self, hidden):
        return hidden + torch.arange(hidden.shape[0], dtype=hidden.dtype).unsqueeze(1)


class Summary(torch.nn.Module):
    """Returns one value for the whole input."""

    def forward(self, hidden):
        return hidden.sum()


class Doubled(torch.nn.Module):
    """Row-wise, but writes its result into the tensor it was given."""

    def forward(self, hidden):
        return hidden.mul_(2.0)


class Gate(torch.nn.Module):
    """Row-wise, of a class the runtime's predicate does not name."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(-1.0, 1.0, WIDTH))

    def forward(self, hidden):
        return torch.relu(hidden * self.weight)


class AddScale(torch.nn.Module):
    """A norm with two call forms, like a fused add-and-normalize: with the
    hidden state alone it returns a new tensor; given a residual as well it
    writes the sum into the residual."""

    def forward(self, hidden, residual=None):
        if residual is None:
            return hidden / hidden.norm(dim=-1, keepdim=True)
        residual.add_(hidden)
        return residual / residual.norm(dim=-1, keepdim=True), residual


def tokenwise(call) -> bool:
    """The toy runtime's predicate: by class, and for `AddScale` by call form."""
    module = call.envoy._module
    if isinstance(module, AddScale):
        return len(call.args) + len(call.kwargs) == 1
    return isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Embedding))


class Scaled(torch.nn.Module):
    def forward(self, hidden, head, scale=1.0, bias=None):
        out = head(hidden) * scale
        return {"scores": out if bias is None else out + bias, "rows": (hidden,)}


class Reader(torch.nn.Module):
    def __init__(self, tuple_output: bool = False) -> None:
        super().__init__()
        self.decoder_blocks = torch.nn.ModuleList(Block(tuple_output) for _ in range(3))
        self.final_scale = torch.nn.LayerNorm(WIDTH)
        self.output_projection = torch.nn.Linear(WIDTH, 20)
        self.row_mean = RowMean()
        self.row_position = RowPosition()
        self.summary = Summary()
        self.scaled = Scaled()
        self.doubled = Doubled()
        self.feature_gate = Gate()
        self.add_scale = AddScale()
        self.token_table = torch.nn.Embedding(5, WIDTH)
        self.side_path = torch.nn.Sequential(torch.nn.Linear(WIDTH, WIDTH))

    def forward(self, hidden):
        for block in self.decoder_blocks:
            hidden = block(hidden)
            hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        return self.output_projection(self.final_scale(hidden))


class Stacked(NNsight):
    def _batch_size(self, *inputs, **kwargs):
        return inputs[0].shape[0] if inputs else 0

    def _batch(self, invokes, fn):
        return (torch.cat([inputs[0] for inputs, _ in invokes]),), {}


def counted(module: torch.nn.Module) -> list:
    """A list that grows by the row count of every call of ``module``."""
    calls: list = []
    module.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[0]))
    return calls


@pytest.fixture(params=[False, True], ids=["tensor-output", "tuple-output"])
def model(request):
    torch.manual_seed(0)
    return Stacked(Reader(request.param).eval())


@pytest.fixture
def inputs():
    torch.manual_seed(1)
    # Ragged row counts, a one-row caller among them.
    return [torch.randn(rows, WIDTH) for rows in (1, 3, 2, 5)]


def hidden_of(output):
    return output[0] if isinstance(output, tuple) else output


def lens(model, inputs, body=None):
    """A logit lens over every block, one invoke per input; ``[per-invoke saves]``.

    Saved by invoke index: with a coalescer the workers take turns at every
    module call, so they do not finish in the order they were written.
    """
    with torch.no_grad(), model.trace() as tracer:
        saved = nnsight.save({})
        for index, rows in enumerate(inputs):
            with tracer.invoke(rows):
                read = []
                for block in model.decoder_blocks:
                    hidden = hidden_of(block.output)
                    read.append(model.output_projection(model.final_scale(hidden)))
                if body is not None:
                    read.append(body(model, index, hidden))
                saved[index] = torch.stack(read)
    return [saved[index] for index in range(len(inputs))]


def both(model, run):
    """``run()`` with every worker calling for itself, then with a coalescer."""
    plain = run()
    model.interleaver.coalescer = Coalescer(tokenwise)
    return plain, run()


def test_a_module_called_by_every_worker_runs_once(model, inputs):
    scale, project = counted(model.final_scale._module), counted(model.output_projection._module)
    plain, shared = both(model, lambda: lens(model, inputs))

    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(plain, shared))
    rows, blocks = sum(len(rows) for rows in inputs), len(model.decoder_blocks)
    # Without a coalescer: a call per invoke per block, and the forward's own.
    assert scale[: len(inputs) * blocks + 1].count(rows) == 1
    # With one: one call of every invoke's rows per block, and nothing else.
    assert scale[len(inputs) * blocks + 1 :] == [rows] * blocks + [rows]
    assert project[len(inputs) * blocks + 1 :] == [rows] * blocks + [rows]


def test_a_block_that_branches_on_its_data_calls_alone(model, inputs):
    # Invokes with an even row count take another path through the block. They
    # are not held back, and nobody waits for them.
    def body(model, index, hidden):
        if hidden.shape[0] % 2 == 0:
            return model.output_projection(model.final_scale(hidden * 2.0))
        return model.output_projection(hidden)

    plain, shared = both(model, lambda: lens(model, inputs, body))
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(plain, shared))


@pytest.mark.parametrize("name", ["row_mean", "row_position", "summary", "feature_gate"])
def test_a_module_the_predicate_does_not_name_runs_alone(model, inputs, name):
    calls = counted(getattr(model, name)._module)

    def body(model, index, hidden):
        out = getattr(model, name)(hidden)
        return model.output_projection(out if out.ndim else hidden * out)

    plain, shared = both(model, lambda: lens(model, inputs, body))
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(plain, shared))
    # One call per invoke each time, none on two invokes' rows.
    assert sorted(calls) == sorted([len(rows) for rows in inputs] * 2)


def test_a_registered_module_is_shared(model, inputs):
    calls = counted(model.feature_gate._module)

    def run():
        return lens(model, inputs, lambda model, index, hidden: model.output_projection(model.feature_gate(hidden)))

    plain = run()
    model.interleaver.coalescer = Coalescer(tokenwise)
    model.interleaver.coalescer.register(model.feature_gate._module)
    del calls[:]
    shared = run()

    assert calls == [sum(len(rows) for rows in inputs)]
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(plain, shared))


def test_a_coalescer_without_a_predicate_shares_only_registered_modules(model, inputs):
    scale, project = counted(model.final_scale._module), counted(model.output_projection._module)
    model.interleaver.coalescer = Coalescer()
    model.interleaver.coalescer.register(model.output_projection._module)
    lens(model, inputs)

    rows, blocks = sum(len(rows) for rows in inputs), len(model.decoder_blocks)
    assert project == [rows] * blocks + [rows]
    assert sorted(scale) == sorted([len(rows) for rows in inputs] * blocks + [rows])


def test_a_norm_is_shared_alone_and_not_with_a_residual_it_writes_into(model, inputs):
    calls = counted(model.add_scale._module)
    model.interleaver.coalescer = Coalescer(tokenwise)
    with torch.no_grad(), model.trace() as tracer:
        saved = nnsight.save({})
        for index, rows in enumerate(inputs):
            with tracer.invoke(rows):
                hidden = hidden_of(model.decoder_blocks[2].output).clone()
                pure = model.add_scale(hidden)
                residual = torch.ones_like(hidden)
                fused, _ = model.add_scale(hidden, residual)
                # The caller's own residual holds the sum.
                saved[index] = (pure, hidden, fused, torch.equal(residual, hidden + 1.0))
    total = sum(len(rows) for rows in inputs)
    # One shared call of the first form, then one call per invoke of the second.
    assert calls[0] == total and sorted(calls[1:]) == sorted(len(rows) for rows in inputs)
    for index in range(len(inputs)):
        pure, hidden, fused, wrote = saved[index]
        assert wrote
        assert torch.allclose(pure, hidden / hidden.norm(dim=-1, keepdim=True), atol=1e-6)
        assert torch.allclose(fused, (hidden + 1.0) / (hidden + 1.0).norm(dim=-1, keepdim=True), atol=1e-6)


def test_an_error_reaches_only_the_worker_whose_call_raised(model, inputs):
    model.interleaver.coalescer = Coalescer(tokenwise)
    model.interleaver.defer_exceptions = True
    try:
        with torch.no_grad(), model.trace() as tracer:
            saved = nnsight.save({})
            for index, rows in enumerate(inputs):
                with tracer.invoke(rows):
                    hidden = hidden_of(model.decoder_blocks[0].output)
                    if index == 1:
                        hidden = hidden[:, :3]  # the norm refuses this width
                    try:
                        saved[index] = model.final_scale(hidden)
                    except RuntimeError as error:
                        saved[index] = "raised"
    finally:
        model.interleaver.defer_exceptions = False
    assert saved[1] == "raised"
    assert all(saved[index].shape == (len(inputs[index]), WIDTH) for index in (0, 2, 3))


def test_an_error_from_one_callers_values_reaches_only_that_caller(model, inputs):
    # Same shapes for every caller, one-dimensional rows; one caller's ids are
    # outside the table, which the call on everyone's rows raises on.
    model.interleaver.coalescer = Coalescer(tokenwise)
    with torch.no_grad(), model.trace() as tracer:
        saved = nnsight.save({})
        for index, rows in enumerate(inputs):
            with tracer.invoke(rows):
                count = hidden_of(model.decoder_blocks[1].output).shape[0]
                ids = torch.full((count,), 9 if index == 2 else index)
                try:
                    saved[index] = model.token_table(ids)
                except IndexError:
                    saved[index] = "raised"
    assert saved[2] == "raised"
    for index in (0, 1, 3):
        assert torch.equal(saved[index], model.token_table.weight[index].expand(len(inputs[index]), -1))


def test_a_module_that_writes_into_its_input_is_never_shared(model, inputs):
    calls = counted(model.doubled._module)
    model.interleaver.coalescer = Coalescer(tokenwise)
    with torch.no_grad(), model.trace() as tracer:
        saved = nnsight.save({})
        for index, rows in enumerate(inputs):
            with tracer.invoke(rows):
                given = hidden_of(model.decoder_blocks[2].output).clone()
                before = given.clone()
                model.doubled(given)
                # The caller's own tensor holds the result.
                saved[index] = torch.equal(given, before * 2.0)
    assert all(saved[index] for index in range(len(inputs)))
    assert sorted(calls) == sorted(len(rows) for rows in inputs)


def test_a_call_before_the_first_read(model, inputs):
    calls = counted(model.final_scale._module)
    model.interleaver.coalescer = Coalescer(tokenwise)
    with torch.no_grad(), model.trace() as tracer:
        saved = nnsight.save({})
        for index, rows in enumerate(inputs):
            with tracer.invoke(rows):
                early = model.final_scale(rows)
                saved[index] = (early, hidden_of(model.decoder_blocks[0].output).clone())
    # One call shared before the forward started, then the model's own call on
    # all rows.
    total = sum(len(rows) for rows in inputs)
    assert calls == [total, total]
    reference = model._module
    for index, rows in enumerate(inputs):
        early, read = saved[index]
        assert torch.allclose(early, reference.final_scale(rows), atol=1e-6)
        assert torch.allclose(read, hidden_of(reference.decoder_blocks[0](rows)), atol=1e-6)


def test_a_watched_call_after_a_shared_call(model, inputs):
    # A worker resumed from its call makes a `hook=True` call of a module
    # another worker is waiting to read.
    calls = counted(model.final_scale._module)
    model.interleaver.coalescer = Coalescer(tokenwise)
    with torch.no_grad(), model.trace() as tracer:
        saved = nnsight.save({})
        with tracer.invoke(inputs[1]):
            scaled = model.final_scale(hidden_of(model.decoder_blocks[0].output))
            saved["caller"] = model.side_path(scaled, hook=True)
        with tracer.invoke(inputs[3]):
            saved["reader"] = model.side_path.output.clone()
    assert torch.equal(saved["caller"], saved["reader"])
    # The caller's one call of the norm ran once, then the model's own.
    assert calls == [3, 8]


def test_arguments_that_are_not_rows(model, inputs):
    calls = counted(model.scaled._module)
    bias = torch.ones(20)

    def body(scale_of):
        def call(model, index, hidden):
            # The head is passed as an envoy, the bias as one shared tensor.
            out = model.scaled(hidden, model.output_projection, scale=scale_of(index), bias=bias)
            assert out["rows"][0].shape == hidden.shape
            return out["scores"]

        return lambda: lens(model, inputs, call)

    plain = body(lambda index: 2.0)()
    model.interleaver.coalescer = Coalescer(tokenwise)
    model.interleaver.coalescer.register(model.scaled._module)
    shared = body(lambda index: 2.0)()
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(plain, shared))
    assert calls[-1] == sum(len(rows) for rows in inputs)

    # A plain argument that differs between workers keeps their calls apart.
    del calls[:]
    apart = body(lambda index: float(index))()
    assert sorted(calls) == sorted(len(rows) for rows in inputs)
    assert not torch.allclose(apart[1], shared[1])


def test_split_gives_each_caller_its_rows():
    rows = torch.arange(6.0).reshape(6, 1)
    first, second = split({"a": rows, "b": (rows, None)}, [2, 4])
    assert torch.equal(first["a"], rows[:2]) and torch.equal(second["b"][0], rows[2:])
    assert second["b"][1] is None
    assert split(None, [2, 4]) == [None, None]
    for result in (torch.tensor(1.0), torch.zeros(5, 1), "text", (rows, 3)):
        with pytest.raises(Unsplittable):
            split(result, [2, 4])


def test_replacements_of_several_workers_at_one_visit(model, inputs):
    def run(blocks):
        with torch.no_grad(), model.trace() as tracer:
            saved = nnsight.save({})
            for index, rows in enumerate(inputs):
                with tracer.invoke(rows):
                    if index in blocks:
                        block = model.decoder_blocks[1]
                        hidden = hidden_of(block.output)
                        new = hidden * (index + 2.0)
                        block.output = (new, None) if isinstance(block.output, tuple) else new
                        # Its own replacement, read back at the same visit.
                        assert torch.equal(hidden_of(block.output), new)
                    saved[index] = model.output.clone()
        return saved

    def alone(index):
        with torch.no_grad(), model.trace(inputs[index]):
            block = model.decoder_blocks[1]
            if index in (0, 2, 3):
                new = hidden_of(block.output) * (index + 2.0)
                block.output = (new, None) if isinstance(block.output, tuple) else new
            out = model.output.clone().save()
        return out

    together = run({0, 2, 3})
    for index in range(len(inputs)):
        assert torch.allclose(together[index], alone(index), atol=1e-6)


def test_the_vllm_batcher_writes_every_replacement_with_the_models_rows_between():
    from nnsight.modeling.vllm.batching import VLLMBatcher

    batcher = VLLMBatcher(None)
    batcher.total = 10
    for slab in (torch.arange(10.0).unsqueeze(1), torch.arange(10.0).reshape(1, 10, 1)):
        axis = 0 if slab.shape[0] == 10 else 1
        running = slab
        # Arriving out of span order, with gaps before, between and after.
        for start, size, fill in ((6, 2, -1.0), (1, 3, -2.0)):
            running = batcher.gather_swap(
                running, [start, size], torch.full_like(slab.narrow(axis, start, size), fill)
            )
        merged = batcher.assemble_swap(running)
        assert merged.shape == slab.shape
        assert merged.flatten().tolist() == [0, -2, -2, -2, 4, 5, -1, -1, 8, 9]


def test_a_replacement_of_another_shape_is_that_workers_error(model, inputs):
    with torch.no_grad(), model.trace() as tracer:
        saved = nnsight.save({})
        for index, rows in enumerate(inputs):
            with tracer.invoke(rows):
                block = model.decoder_blocks[1]
                hidden = hidden_of(block.output)
                new = hidden[:1] if index == 1 else hidden + 1.0
                try:
                    block.output = (new, None) if isinstance(block.output, tuple) else new
                except ValueError as error:
                    saved["refused"] = str(error)
                saved[index] = model.output.clone()

    assert "keep the shape of the rows it replaces" in saved["refused"]
    with torch.no_grad(), model.trace(inputs[1]):
        untouched = model.output.clone().save()
    assert torch.allclose(saved[1], untouched, atol=1e-6)
    assert not torch.allclose(saved[2], model._module(inputs[2]), atol=1e-6)


def test_a_replacement_of_another_container_is_that_workers_error(inputs):
    torch.manual_seed(0)
    model = Stacked(Reader(tuple_output=True).eval())
    with torch.no_grad(), model.trace() as tracer:
        saved = nnsight.save({})
        for index, rows in enumerate(inputs):
            with tracer.invoke(rows):
                block = model.decoder_blocks[1]
                new = block.output[0] + 1.0
                try:
                    block.output = (new,) if index == 1 else (new, None)
                except ValueError as error:
                    saved["refused"] = str(error)
                saved[index] = model.output.clone()
    assert "a sequence of 2" in saved["refused"]
    assert torch.allclose(saved[1], model._module(inputs[1]), atol=1e-6)
    assert not torch.allclose(saved[3], model._module(inputs[3]), atol=1e-6)
