"""
Patch for python-engineio to fix SSL race condition during WebSocket handshake.

This module must be imported BEFORE any socketio imports to ensure the patch
is applied before any connections are made.

Problem:
    When connecting via WebSocket over SSL/TLS, engineio starts background
    threads immediately after the 'connect' event. The Socket.IO CONNECT
    packet is sent via the write thread while the read thread simultaneously
    waits for the ACK. This concurrent send/recv corrupts the SSL socket
    because Python's SSL sockets are not thread-safe for simultaneous
    read+write operations.

Solution:
    Flush queued packets and receive the response synchronously in the main
    thread before starting background threads.

See: https://github.com/miguelgrinberg/python-socketio/issues/1568
"""

_patched = False


def apply_patch():
    """
    Apply the SSL handshake race condition fix to engineio.

    This function is idempotent - calling it multiple times has no effect.
    """
    global _patched
    if _patched:
        return

    import engineio.client
    from engineio import packet as eio_packet

    _original_connect_websocket = engineio.client.Client._connect_websocket

    def _patched_connect_websocket(self, url, headers, engineio_path):
        """
        Patched version of _connect_websocket that serializes the handshake.
        """
        try:
            import websocket
        except ImportError:
            return _original_connect_websocket(self, url, headers, engineio_path)

        # For upgrades (sid already exists), use original behavior
        if self.sid:
            return _original_connect_websocket(self, url, headers, engineio_path)

        # === Begin: copied setup from original _connect_websocket ===
        import ssl
        from base64 import b64encode
        from http.cookies import SimpleCookie
        import urllib.parse

        websocket_url = self._get_engineio_url(url, engineio_path, 'websocket')
        self.base_url = websocket_url
        self.logger.info('Attempting WebSocket connection to ' + websocket_url)

        cookies = None
        extra_options = {}

        if self.http:
            ck = SimpleCookie()
            for cookie in self.http.cookies:
                ck[cookie.name] = cookie.value
            cookies = ck.output(header='', sep=';').strip()
            for header, value in list(headers.items()):
                if header.lower() == 'cookie':
                    if cookies:
                        cookies += '; '
                    cookies += value
                    del headers[header]
                    break

            if 'Authorization' not in headers and self.http.auth is not None:
                if not isinstance(self.http.auth, tuple):
                    raise ValueError('Only basic authentication is supported')
                basic_auth = '{}:{}'.format(
                    self.http.auth[0], self.http.auth[1]).encode('utf-8')
                basic_auth = b64encode(basic_auth).decode('utf-8')
                headers['Authorization'] = 'Basic ' + basic_auth

            if isinstance(self.http.cert, tuple):
                extra_options['sslopt'] = {
                    'certfile': self.http.cert[0],
                    'keyfile': self.http.cert[1]}
            elif self.http.cert:
                extra_options['sslopt'] = {'certfile': self.http.cert}

            if self.http.proxies:
                proxy_url = None
                if websocket_url.startswith('ws://'):
                    proxy_url = self.http.proxies.get(
                        'ws', self.http.proxies.get('http'))
                else:
                    proxy_url = self.http.proxies.get(
                        'wss', self.http.proxies.get('https'))
                if proxy_url:
                    parsed_url = urllib.parse.urlparse(
                        proxy_url if '://' in proxy_url else 'scheme://' + proxy_url)
                    extra_options['http_proxy_host'] = parsed_url.hostname
                    extra_options['http_proxy_port'] = parsed_url.port
                    extra_options['http_proxy_auth'] = (
                        (parsed_url.username, parsed_url.password)
                        if parsed_url.username or parsed_url.password else None)

            if isinstance(self.http.verify, str):
                if 'sslopt' in extra_options:
                    extra_options['sslopt']['ca_certs'] = self.http.verify
                else:
                    extra_options['sslopt'] = {'ca_certs': self.http.verify}
            elif not self.http.verify:
                self.ssl_verify = False

        if not self.ssl_verify:
            if 'sslopt' in extra_options:
                extra_options['sslopt'].update({"cert_reqs": ssl.CERT_NONE})
            else:
                extra_options['sslopt'] = {"cert_reqs": ssl.CERT_NONE}

        headers.update(self.websocket_extra_options.pop('header', {}))
        extra_options['header'] = headers
        extra_options['cookie'] = cookies
        extra_options['enable_multithread'] = True
        extra_options['timeout'] = self.request_timeout
        extra_options.update(self.websocket_extra_options)

        try:
            ws = websocket.create_connection(
                websocket_url + self._get_url_timestamp(), **extra_options)
        except (ConnectionError, OSError, websocket.WebSocketException):
            raise engineio.exceptions.ConnectionError('Connection error')

        # Receive Engine.IO OPEN packet
        try:
            p = ws.recv()
        except Exception as e:
            raise engineio.exceptions.ConnectionError(
                'Unexpected recv exception: ' + str(e))

        open_packet = eio_packet.Packet(encoded_packet=p)
        if open_packet.packet_type != eio_packet.OPEN:
            raise engineio.exceptions.ConnectionError('no OPEN packet')

        self.logger.info(
            'WebSocket connection accepted with ' + str(open_packet.data))
        self.sid = open_packet.data['sid']
        self.upgrades = open_packet.data['upgrades']
        self.ping_interval = int(open_packet.data['pingInterval']) / 1000.0
        self.ping_timeout = int(open_packet.data['pingTimeout']) / 1000.0
        self.current_transport = 'websocket'

        self.state = 'connected'
        engineio.base_client.connected_clients.append(self)
        self._trigger_event('connect', run_async=False)
        # === End: copied setup ===

        # === Begin: THE FIX ===
        # Flush any packets queued by the connect event handler (e.g.,
        # Socket.IO CONNECT) and receive the response synchronously.
        # This avoids a race condition where concurrent send/recv on SSL
        # sockets can corrupt the connection.
        try:
            while True:
                pkt = self.queue.get_nowait()
                if pkt is None:
                    self.queue.task_done()
                    break
                self.logger.info(
                    'Sending packet %s data %s (during handshake)',
                    eio_packet.packet_names[pkt.packet_type],
                    pkt.data if not isinstance(pkt.data, bytes) else '<binary>')
                if pkt.binary:
                    ws.send_binary(pkt.encode())
                else:
                    ws.send(pkt.encode())
                self.queue.task_done()
        except self.queue_empty:
            pass

        # Receive the response (e.g., Socket.IO CONNECT ACK) synchronously
        try:
            p = ws.recv()
            if p:
                pkt = eio_packet.Packet(encoded_packet=p)
                self._receive_packet(pkt)
        except Exception:
            # If recv fails here, the read loop will handle it
            pass
        # === End: THE FIX ===

        self.ws = ws
        self.ws.settimeout(self.ping_interval + self.ping_timeout)

        # Start background threads (now safe - handshake is complete)
        self.write_loop_task = self.start_background_task(self._write_loop)
        self.read_loop_task = self.start_background_task(
            self._read_loop_websocket)
        return True

    # Apply the patch
    engineio.client.Client._connect_websocket = _patched_connect_websocket
    _patched = True


# Auto-apply patch on import
apply_patch()
