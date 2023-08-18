if __name__ == "__main__":
    # Pytorch/cuda requires 'spawn' instead of 'fork' for multiprocessing
    import multiprocessing

    multiprocessing.set_start_method("spawn")

    from . import CONFIG, download_models
    from .app import app, socketio_app

    socketio_app.run(
        app, host="0.0.0.0", port=CONFIG.PORT, debug=True, use_reloader=False
    )
