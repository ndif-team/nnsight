from nnsight import DiffuserModel


diffuser = DiffuserModel("CompVis/stable-diffusion-v1-4", device='cuda:0')


with diffuser.generate() as generator:

    with generator.invoke(["Blue elephant", "GREEN"]) as invoker:

        pass
diffuser.to_image(generator.output)[0].save('ummb.png')
diffuser.to_image(generator.output)[1].save('umm.png')