from engine import Model
import torch

model = Model('gpt2')



with model.optimize() as invoker:

    soft_prompt = 


    for target, data in dataset:

        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()