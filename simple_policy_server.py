"""
Simple Policy Server for BEHAVIOR Environment

This is a minimal WebSocket server that implements a dummy policy returning zero actions.
It is useful for testing the environment connection and data flow without running heavy models.
"""

import argparse
import asyncio
import functools
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import msgpack
import numpy as np
import websockets
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("simple_policy_server")

# NumPy array support for msgpack
def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj

def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj

packb = functools.partial(msgpack.packb, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


class SimplePolicy:
    """
    A minimal policy that returns zero actions.
    """
    def __init__(self, config: Dict[str, Any]):
        self.action_dim = config["policy"]["action_dim"]
        logger.info(f"Initialized SimplePolicy with action_dim={self.action_dim}")

    def reset(self) -> None:
        logger.info("Policy reset")

    def predict(self, obs: Dict[str, Any]) -> np.ndarray:
        # Return zero action
        print("SimplePolicy.predict called")
        logger.info("Policy predict called")
        return np.zeros(self.action_dim, dtype=np.float32)


async def handle_client(websocket, policy: SimplePolicy):
    logger.info(f"Client connected: {websocket.remote_address}")
    try:
        # Send metadata
        metadata = {"action_dim": policy.action_dim, "server": "simple_policy_server"}
        await websocket.send(packb(metadata))

        async for message in websocket:
            try:
                obs = unpackb(message)
                if isinstance(obs, dict) and obs.get("reset", False):
                    policy.reset()
                    await websocket.send(packb({"status": "reset_ok"}))
                    continue
                
                action = policy.predict(obs)
                await websocket.send(packb({"action": action}))
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send(packb({"error": str(e)}))

    except Exception as e:
        logger.error(f"Connection error: {e}")
    finally:
        logger.info(f"Client disconnected: {websocket.remote_address}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="behavior.yaml")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.port:
        config["server"]["port"] = args.port

    policy = SimplePolicy(config)
    
    host = config["server"]["host"]
    port = config["server"]["port"]
    
    logger.info(f"Starting Simple Policy Server at ws://{host}:{port}")

    # HTTP Handler for health checks
    def process_request(conn, req):
        from websockets.http11 import Response
        from websockets.datastructures import Headers
        
        if req.path == "/healthz":
            return Response(200, "OK", Headers([("Content-Type", "text/plain")]), b"OK\n")
        
        if "upgrade" not in req.headers.get("Connection", "").lower():
            return Response(426, "Upgrade Required", Headers([("Content-Type", "text/plain")]), b"Upgrade Required\n")

    async with websockets.serve(
        lambda ws: handle_client(ws, policy),
        host, port,
        process_request=process_request,
        max_size=100*1024*1024
    ):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
