# Start Dev Async Update Summary

## Purpose
Updated `start_dev.py` to run the FastAPI server asynchronously using the pattern provided, with enhanced command-line argument support for different deployment modes.

## Changes Made

### **1. Added Async Server Functions**

#### **`run_service_async()`**
```python
async def run_service_async(
    host: str = "0.0.0.0", 
    port: int = 8000, 
    workers: int = 1,
    reload: bool = True,
    log_level: str = "info"
):
    """Run FastAPI service safely within an existing event loop."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting medAI MVP Service on {host}:{port}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Reload: {reload}")
    logger.info(f"Log level: {log_level}")

    config_uvicorn = uvicorn.Config(
        "src.api.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=reload,
    )
    server = uvicorn.Server(config_uvicorn)
    await server.serve()
```

#### **`run_production_service()`**
```python
async def run_production_service(
    host: str = "0.0.0.0", 
    port: int = 8000, 
    workers: int = 1,
    log_level: str = "info"
):
    """Run FastAPI service in production mode (no reload)."""
    return await run_service_async(
        host=host,
        port=port,
        workers=workers,
        reload=False,
        log_level=log_level
    )
```

### **2. Added Command-Line Arguments**

#### **`parse_arguments()`**
```python
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="medAI MVP Development Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--production", action="store_true", help="Run in production mode (no reload)")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="Log level")
    parser.add_argument("--skip-checks", action="store_true", help="Skip environment and microservice checks")
    return parser.parse_args()
```

### **3. Updated Main Function**

#### **Enhanced `main()`**
- **Argument parsing**: Supports command-line configuration
- **Mode selection**: Development vs Production modes
- **Async execution**: Uses `asyncio.run()` for async server startup
- **Error handling**: Proper exception handling and logging

### **4. Removed Legacy Code**
- **Removed**: Old `start_server()` function
- **Simplified**: Direct async execution in main function

## Usage Examples

### **Development Mode (Default)**
```bash
python start_dev.py
```
- Host: 0.0.0.0
- Port: 8000
- Workers: 1
- Reload: True
- Log Level: info

### **Production Mode**
```bash
python start_dev.py --production --workers 4 --log-level warning
```
- Host: 0.0.0.0
- Port: 8000
- Workers: 4
- Reload: False
- Log Level: warning

### **Custom Configuration**
```bash
python start_dev.py --host 127.0.0.1 --port 9000 --workers 2 --log-level debug
```
- Host: 127.0.0.1
- Port: 9000
- Workers: 2
- Reload: True
- Log Level: debug

### **Skip Checks (Fast Startup)**
```bash
python start_dev.py --skip-checks
```
- Skips environment and microservice checks
- Faster startup for development

## Benefits

### **Async Architecture** ⚡
- **Better performance**: Async server startup and management
- **Event loop integration**: Proper async/await pattern
- **Resource efficiency**: Better memory and CPU usage

### **Flexibility** 🔧
- **Command-line configuration**: Easy deployment customization
- **Multiple modes**: Development and production configurations
- **Worker scaling**: Support for multiple worker processes

### **Production Ready** 🚀
- **Production mode**: No reload for production deployments
- **Worker processes**: Support for multiple workers
- **Logging levels**: Configurable logging for different environments
- **Error handling**: Robust exception handling

### **Development Friendly** 🛠️
- **Hot reload**: Development mode with auto-reload
- **Skip checks**: Fast startup for development
- **Debug logging**: Detailed logging for troubleshooting

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | 0.0.0.0 | Host to bind to |
| `--port` | 8000 | Port to bind to |
| `--workers` | 1 | Number of worker processes |
| `--production` | False | Run in production mode (no reload) |
| `--log-level` | info | Log level (debug, info, warning, error) |
| `--skip-checks` | False | Skip environment and microservice checks |

## Files Modified
- `start_dev.py` - Complete rewrite with async server functions and command-line arguments

## Conclusion
The `start_dev.py` script now uses a modern async architecture with comprehensive command-line argument support, making it suitable for both development and production deployments while maintaining the same functionality as before.
