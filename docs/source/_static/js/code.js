var simple = `with model.trace() as tracer:
  
with tracer.invoke(input2):
  l2_input = model.layer2.input
  
with tracer.invoke(input1):
  model.layer2.input = l2_input
  output = model.layer2.output.save()

print(output)`;

var trace = `with model.trace() as tracer:
  
with tracer.invoke(input2):
  l2_input = model.layer2.input
  
with tracer.invoke(input1):
  model.layer2.input = l2_input
  output = model.layer2.output.save()

print(output)`;

var generate = `with model.generate() as tracer:
  
with tracer.invoke(input2):
  l2_input = model.layer2.input

print(output)`;

document.addEventListener('DOMContentLoaded', (event) => {
    document.querySelectorAll('code.language-python.simple').forEach(el => {
        el.textContent = simple;
    });
    document.querySelectorAll('code.language-python.trace').forEach(el => {
        el.textContent = trace;
    });
    document.querySelectorAll('code.language-python.generate').forEach(el => {
        el.textContent = generate;
    });
});
