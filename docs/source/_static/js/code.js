var simple = `from nnsight import NNsight, LanguageModel

net = torch.nn.Sequential(OrderedDict([
  ('layer1', torch.nn.Linear(input_size, hidden_dims)),
  ('layer2', torch.nn.Linear(hidden_dims, output_size)),
]))

model = NNsight(net)

...

model = LanguageModel('openai-community/gpt2')`;

var trace = `with model.trace('Who invented neural networks?'):

  hidden_state_output = model.layer1.output.save()
  hidden_state_input = model.layer2.input.save()

  output = model.output.save()
  
print(hidden_state_output)
print(hidden_state_input)
print(output)`;

var multi = `with model.trace() as tracer:
  
  with tracer.invoke('The Eiffel Tower is in the city of'):

    model.transformer.h[-1].mlp.output[0][:] = 0

    intervention = model.lm_head.output.argmax(dim=-1).save()

  with tracer.invoke('The Eiffel Tower is in the city of'):

    original = model.lm_head.output.argmax(dim=-1).save()

print(output)`;

document.addEventListener('DOMContentLoaded', (event) => {
    document.querySelectorAll('code.language-python.simple').forEach(el => {
        el.textContent = simple;
    });
    document.querySelectorAll('code.language-python.trace').forEach(el => {
        el.textContent = trace;
    });
    document.querySelectorAll('code.language-python.multi').forEach(el => {
        el.textContent = multi;
    });
    document.querySelectorAll('div.output_area.stderr').forEach(el => {
        el.style.visibility = "hidden";
        el.style.display = "none";
    });
    document.querySelectorAll('div.output_area.stderr').forEach(el => {
      el.style.visibility = "hidden";
      el.style.display = "none";
    });
    document.querySelectorAll('div.output_area').forEach(el => {
      el.style.marginBottom = "1em";
    });
    document.querySelectorAll('div.input_area').forEach(el => {
      el.style.border = "0px";
      el.style.marginTop = "0.5em";
      el.style.marginBottom = "0.5em";
    });
    document.querySelectorAll('summary').forEach(el => {
      el.style.marginTop = "1em";
      el.style.marginBottom = "1em";
      document.querySelectorAll('p').forEach(p => {
          const span = document.createElement('span');

          // Copy the innerHTML from the <p> to the <span>
          span.innerHTML = p.innerHTML;

          // Optional: Copy styles from the <p> to the <span>
          span.style.cssText = p.style.cssText;

          // Replace the <p> with the <span> in the DOM
          p.parentNode.replaceChild(span, p);
        });
  });
});

