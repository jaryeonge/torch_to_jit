#include <iostream>
#include <torch/script.h>
#include <memory>

int main(int argc, const char* argv[])
{
	if (argc != 3) {
		std::cerr << "usage: example-app <path-to-exported-script-module> <path-to-exported-tokenizer>\n";
		return -1;
	}
	
	// model
	torch::jit::script::Module module;
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		module = torch::jit::load(argv[1]);
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}
	std::cout << "model is loaded\n";

	// model test
	torch::Tensor a = torch::randint(10, {1, 10});

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(at::_cast_Long(a));

	std::cout << inputs << std::endl;

	auto outputs = module.forward(inputs);

	auto last_hidden_state = outputs.toTuple()->elements()[0].toTensor();
	auto pooler_output = outputs.toTuple()->elements()[1].toTensor();
	std::cout << last_hidden_state << std::endl;
	std::cout << pooler_output << std::endl;
}

