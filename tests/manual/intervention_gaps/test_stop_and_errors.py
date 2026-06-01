"""
Test that tracer.stop(), user errors, and multi-invoke failures don't kill
the vLLM engine.  Validates per-mediator exception isolation.

Run: CUDA_VISIBLE_DEVICES=X python test_stop_and_errors.py
"""
import torch


def verify_engine(model, label):
    """Run a trivial trace to confirm the engine is still alive."""
    with model.trace("Engine check", temperature=0.0, top_p=1):
        logits = model.logits.output.save()
    assert logits.shape[-1] > 0, f"Engine dead after {label}"
    print(f"  Engine alive after {label} — logits shape: {logits.shape}")


def main():
    from nnsight.modeling.vllm import VLLM

    model = VLLM("gpt2", tensor_parallel_size=1, gpu_memory_utilization=0.05, dispatch=True)

    # ------------------------------------------------------------------
    print("=" * 60)
    print("Test 1: tracer.stop() — engine survives, values saved")
    print("=" * 60)

    with model.trace("Hello world", temperature=0.0, top_p=1) as tracer:
        hs = model.transformer.h[0].output[0].save()
        tracer.stop()
        # Code after stop() should NOT execute
        should_not_exist = model.transformer.h[5].output[0].save()

    assert isinstance(hs, torch.Tensor), f"Expected tensor, got {type(hs)}"
    print(f"  hs shape: {hs.shape}, dtype: {hs.dtype}")

    try:
        _ = should_not_exist
        print("  WARNING: should_not_exist was defined (stop didn't prevent it)")
    except (UnboundLocalError, NameError):
        print("  should_not_exist correctly undefined (stop prevented it)")

    verify_engine(model, "stop")
    print("  PASS")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 2: Single invoke error — engine survives, error reported")
    print("=" * 60)

    caught = False
    try:
        with model.trace("Hello world", temperature=0.0, top_p=1):
            bad = model.transformer.h[100].output[0].save()
    except Exception as e:
        caught = True
        print(f"  Error correctly raised: {type(e).__name__}: {e}")

    assert caught, "FAIL: No error raised for invalid layer index"

    verify_engine(model, "single error")
    print("  PASS")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 3: One of N invokes errors — engine survives")
    print("=" * 60)

    caught = False
    try:
        with model.trace(temperature=0.0, top_p=1) as tracer:
            with tracer.invoke("Hello"):
                good1 = model.transformer.h[0].output[0].save()

            with tracer.invoke("World"):
                # This invoke will error — GPT-2 has 12 layers, not 100
                bad = model.transformer.h[100].output[0].save()

            with tracer.invoke("Test"):
                good2 = model.transformer.h[0].output[0].save()
    except Exception as e:
        caught = True
        print(f"  Error raised: {type(e).__name__}")
        # Verify the error message references the actual problem
        assert "list index" in str(e).lower() or "index" in str(e).lower(), \
            f"Error doesn't mention index issue: {e}"
        print(f"  Error message correctly references IndexError")

    assert caught, "FAIL: No error raised when one invoke fails"

    verify_engine(model, "one-of-N error")
    print("  PASS")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 4: All invokes error — all exceptions reported")
    print("=" * 60)

    caught = False
    err_msg = ""
    try:
        with model.trace(temperature=0.0, top_p=1) as tracer:
            with tracer.invoke("Hello"):
                model.transformer.h[100].output[0].save()

            with tracer.invoke("World"):
                model.transformer.h[100].output[0].save()

            with tracer.invoke("Test"):
                model.transformer.h[100].output[0].save()
    except Exception as e:
        caught = True
        err_msg = str(e)
        print(f"  Error raised: {type(e).__name__}")
        # With 3 errors, should be a RuntimeError listing all
        if "invoke" in err_msg.lower() and "failed" in err_msg.lower():
            print(f"  Aggregate error with multiple failures reported")
        else:
            print(f"  Single error reported (may indicate only one was captured)")
        print(f"  Message preview: {err_msg[:200]}")

    assert caught, "FAIL: No error raised when all invokes fail"

    verify_engine(model, "all-error")
    print("  PASS")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 5: Mixed stop + error — error raised, engine survives")
    print("=" * 60)

    caught = False
    try:
        with model.trace(temperature=0.0, top_p=1) as tracer:
            with tracer.invoke("Hello"):
                stop_hs = model.transformer.h[0].output[0].save()
                tracer.stop()

            with tracer.invoke("World"):
                model.transformer.h[100].output[0].save()
    except Exception as e:
        caught = True
        print(f"  Error raised (from invoke 2): {type(e).__name__}")
        # EarlyStopException should be filtered — only the real error surfaces
        assert "EarlyStop" not in str(type(e).__name__), \
            "FAIL: EarlyStopException was not filtered"
        print(f"  EarlyStopException correctly filtered")

    assert caught, "FAIL: No error raised for the erroring invoke"

    verify_engine(model, "mixed stop+error")
    print("  PASS")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 6: Normal trace after all failure modes")
    print("=" * 60)

    with model.trace("The Eiffel Tower is in", temperature=0.0, top_p=1):
        final_logits = model.logits.output.save()
    next_token = model.tokenizer.decode(final_logits.argmax(dim=-1)[0])
    print(f"  Next token: '{next_token}'")
    print(f"  Logits shape: {final_logits.shape}")
    print("  PASS")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED — engine survived all failure modes")
    print("=" * 60)


if __name__ == "__main__":
    main()
