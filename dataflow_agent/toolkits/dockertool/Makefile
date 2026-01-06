IMAGE ?= myorg/dataflow-py:3.11

.PHONY: build image run-thirdparty run-stdlib clean

build:
	python -m dataflow.dataflowagent.toolkits.dockertool.build_docker -i $(IMAGE)

image: build
	docker images | rg $(IMAGE) || true

run-thirdparty:
	python -m dataflow.dataflowagent.toolkits.dockertool.mini_docker \
		--file dataflow/dataflowagent/toolkits/dockertool/examples/hello_thirdparty.py \
		--requirements dataflow/dataflowagent/toolkits/dockertool/examples/requirements_thirdparty.txt \
		--image $(IMAGE) \
		--save dataflow/dataflowagent/toolkits/dockertool/artifacts/hello_thirdparty.tar \
		--timeout 180

run-stdlib:
	python -m dataflow.dataflowagent.toolkits.dockertool.mini_docker \
		--file dataflow/dataflowagent/toolkits/dockertool/examples/hello_stdlib.py \
		--image $(IMAGE) \
		--save dataflow/dataflowagent/toolkits/dockertool/artifacts/hello_stdlib.tar \
		--timeout 120

clean:
	rm -f dataflow/dataflowagent/toolkits/dockertool/artifacts/*.tar
	rm -f dataflow/dataflowagent/toolkits/dockertool/artifacts/run_logs/*

