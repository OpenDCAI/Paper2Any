"""
Test ML services end-to-end flow.

Run mock server first:
    uvicorn dataflow_agent.toolkits.ml_services.mock_server:app --port 8000
"""

import asyncio
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_all_services():
    """Test all ML service clients against mock server."""
    from dataflow_agent.toolkits.ml_services import (
        MinerUClient,
        SAMClient,
        YOLOClient,
        OCRClient,
        RMBGClient,
    )

    base_url = os.environ.get("ML_SERVICE_URL", "http://localhost:8000")
    api_key = os.environ.get("ML_SERVICE_API_KEY", None)

    print(f"Testing against: {base_url}")
    print("-" * 50)

    # Create test image
    test_image = "/tmp/test_ml_services.png"
    try:
        from PIL import Image
        import numpy as np
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(arr).save(test_image)
        print(f"✓ Created test image: {test_image}")
    except ImportError:
        print("✗ PIL not installed, skipping image creation")
        return False

    results = {}

    # Test MinerU
    print("\n[1/5] Testing MinerU...")
    try:
        client = MinerUClient(base_url, api_key=api_key, timeout=10)
        response = await client.parse_image(test_image)
        if response.success and response.blocks:
            print(f"  ✓ MinerU: {len(response.blocks)} blocks parsed")
            results["mineru"] = True
        else:
            print(f"  ✗ MinerU failed: {response.error}")
            results["mineru"] = False
        await client.close()
    except Exception as e:
        print(f"  ✗ MinerU error: {e}")
        results["mineru"] = False

    # Test SAM
    print("\n[2/5] Testing SAM...")
    try:
        client = SAMClient(base_url, api_key=api_key, timeout=10)
        response = await client.segment_auto(test_image, top_k=3)
        if response.success:
            print(f"  ✓ SAM: {len(response.items)} segments found")
            results["sam"] = True
        else:
            print(f"  ✗ SAM failed: {response.error}")
            results["sam"] = False
        await client.close()
    except Exception as e:
        print(f"  ✗ SAM error: {e}")
        results["sam"] = False

    # Test YOLO
    print("\n[3/5] Testing YOLO...")
    try:
        client = YOLOClient(base_url, api_key=api_key, timeout=10)
        response = await client.segment(test_image)
        if response.success:
            labels = [it.label for it in response.items if it.label]
            print(f"  ✓ YOLO: {len(response.items)} objects - {labels}")
            results["yolo"] = True
        else:
            print(f"  ✗ YOLO failed: {response.error}")
            results["yolo"] = False
        await client.close()
    except Exception as e:
        print(f"  ✗ YOLO error: {e}")
        results["yolo"] = False

    # Test OCR
    print("\n[4/5] Testing OCR...")
    try:
        client = OCRClient(base_url, api_key=api_key, timeout=10)
        response = await client.recognize(test_image)
        if response.success:
            texts = [line.text for line in response.lines[:3]]
            print(f"  ✓ OCR: {len(response.lines)} lines - {texts}")
            results["ocr"] = True
        else:
            print(f"  ✗ OCR failed: {response.error}")
            results["ocr"] = False
        await client.close()
    except Exception as e:
        print(f"  ✗ OCR error: {e}")
        results["ocr"] = False

    # Test RMBG
    print("\n[5/5] Testing RMBG...")
    try:
        client = RMBGClient(base_url, api_key=api_key, timeout=10)
        response = await client.remove_background(test_image)
        if response.success and response.image_base64:
            print(f"  ✓ RMBG: output size {response.original_size}")
            results["rmbg"] = True
        else:
            print(f"  ✗ RMBG failed: {response.error}")
            results["rmbg"] = False
        await client.close()
    except Exception as e:
        print(f"  ✗ RMBG error: {e}")
        results["rmbg"] = False

    # Cleanup
    os.unlink(test_image)

    # Summary
    print("\n" + "=" * 50)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Results: {passed}/{total} services passed")

    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")

    return all(results.values())


if __name__ == "__main__":
    success = asyncio.run(test_all_services())
    sys.exit(0 if success else 1)
