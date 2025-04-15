import asyncio

async def inner(future_in, future_out):
    print("Inner: Starting execution")
    # Receive value from outer
    value = await future_in[0]
    print(f"Inner: Received {value} from outer")
    
    # Send value back to outer
    future_out[0].set_result("Hello from inner (1)")
    
    # Receive second value from outer
    value = await future_in[1]
    print(f"Inner: Received {value} from outer")
    
    # Send another value back to outer
    future_out[1].set_result("Hello from inner (2)")
    
    # Receive third value from outer
    value = await future_in[2]
    print(f"Inner: Received {value} from outer")
    
    # Send final value back to outer
    future_out[2].set_result("Hello from inner (3)")
    
    print("Inner: Finished execution")

async def outer():
    print("Outer: Starting execution")
    
    # Create futures for bidirectional communication
    futures_to_inner = [asyncio.Future() for _ in range(3)]
    futures_from_inner = [asyncio.Future() for _ in range(3)]
    
    # Start inner task with futures
    inner_task = asyncio.create_task(inner(futures_to_inner, futures_from_inner))
    
    # Send value to inner
    futures_to_inner[0].set_result("Hello from outer (1)")
    
    # Receive value from inner
    value = await futures_from_inner[0]
    print(f"Outer: Received {value}")
    
    # Send second value to inner
    futures_to_inner[1].set_result("Hello from outer (2)")
    
    # Receive second value from inner
    value = await futures_from_inner[1]
    print(f"Outer: Received {value}")
    
    # Send third value to inner
    futures_to_inner[2].set_result("Hello from outer (3)")
    
    # Receive final value from inner
    value = await futures_from_inner[2]
    print(f"Outer: Received {value}")
    
    await inner_task  # Wait for inner to complete
    print("Outer: Finished execution")

# Run the event loop
if __name__ == "__main__":
    asyncio.run(outer())
