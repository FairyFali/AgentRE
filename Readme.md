This is the official code for the paper: 
> AGENT-REINFORCE: SEARCHING COMPUTE-OPTIMAL MULTI-LLM COLLABORATION GRAPH FOR TEST-TIME SCALING

# Prepare the data
We have prepared the data and put the datasets in the folder [datasets](./datasets).

# Run our code
To run our code, first install the required conda environment.
```bash
conda env create -f environment.yml
```

We provide the commands to reproduce results from our paper below.
```bash
bash experiments/run.sh
```

# Acknowledge
Our code is based on [GPTSwarm](https://github.com/metauto-ai/gptswarm).
