# torch_to_jit

## model transformation
    pip install -r requirements.txt
    
    python __init__.py -p jit -m model_path -n model_name

## C++ test
    ./c_test/build/torchtest-app ./jit_model.pt
