from mineru_tool import run_two_step_extract
from pprint import pprint

image_path = "/home/ubuntu/ziyi/DataFlow-Agent/outputs/paper2expfigure_20251211_163022/pdf_images/page_20.png"
result = run_two_step_extract(image_path, 8001)
pprint(result)
