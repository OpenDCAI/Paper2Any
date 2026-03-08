我打算优化Paper2Figure的pipeline，目前paperagent/pzw/Paper2Any/dataflow_agent/workflow/wf_paper2figure_image_only.py已经实现了输入PDF或描述，调用生图模型生成图像。目前暂时只考虑输入的是绘图需求描述的情况，暂不考虑输入PDF。我想做的优化是：
1. 分析绘图需求描述，调用web search工具（主要应该用arXiv API接口），检索与需求最相关的paper并下载，使用minerU解析下载得到的PDF，抽取出paper中的图，调用vlm对图像做结构化描述和分析，以及Paper的idea/method描述。
2. 步骤1完成后，调用LLM分析检索到的相关paper的内容（Method+Figure）对于当前的绘图需求有哪些关联，有哪些参考价值等
3. 结合步骤2分析得到的关联性和参考价值，根据相关Figure的结构化描述和Idea关联性分析，生成用于vlm生图的prompt，并可以把相关Figures作为参考图一并输入
4. 得到图片后，用一个critic model评判分析，如果需要优化：若结构整体都不太行，或问题比较大，则回到步骤1检索更多相关内容进行参考；若只是局部需要优化，则根据feedback回到步骤3，调整绘图prompt重新做image generation，或生成图片的优化建议，做image edit。

上述流程中涉及到的一些工具，例如web search，minerU解析等，在paperagent/pzw/Paper2Any/dataflow_agent/toolkits中应该都有造好的轮子，你可以优先复用已有的工具和相关代码。在当前项目中没找到相关代码时，才考虑写新代码。

按照上述流程，写一个workflow，存储在paperagent/pzw/Paper2Any/dataflow_agent/workflow，并写一个测试脚本，存储在paperagent/pzw/Paper2Any/tests。
文本的处理和分析，可以用的URL和key为：http://123.129.219.111:3000/v1，sk-yBIfI1TcbftVVFy2uLNfvLRQxE9Z4WFjXEfBQbo2rP8lIDqO。测试时，使用gpt-4o-mini模型。
图像生成和图像编辑，可以用的url和key为：https://api.apiyi.com/v1，sk-1XOECzAbmXWmUplV2f79Eb22E1014cFaA04e42B0A3B6F95d。测试时，使用gemini-3.1-flash-image-preview。

你现在只需要帮我完成workflow的开发并完成测试脚本通过。

整个workflow的中间产物，都需要存储到output路径下，包括Input，search到的相关Paper，解析后的paper相关内容等。

测试和开发时，使用pzw-dev这个conda环境。