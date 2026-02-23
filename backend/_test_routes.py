"""Quick diagnostic: can trading_routes be imported?"""
import sys
sys.path.insert(0, ".")

result_file = open("_test_result.txt", "w")

try:
    from routes.trading_routes import router
    result_file.write(f"SUCCESS: Router loaded with {len(router.routes)} routes\n")
    for route in router.routes:
        path = getattr(route, 'path', '?')
        methods = getattr(route, 'methods', set())
        result_file.write(f"  {methods} {path}\n")
except Exception as e:
    import traceback
    result_file.write(f"IMPORT ERROR: {e}\n")
    result_file.write(traceback.format_exc())

result_file.close()
