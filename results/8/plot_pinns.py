import matplotlib.pyplot as plt

with open("pinn1_epochs.txt", "r") as p1:
    p1_list = [float(i.strip()) for i in p1.readlines()]
    pinn1 = p1_list[1::2]
    p1.close()

with open("pinn2_epochs.txt", "r") as p2:
    p2_list = [float(i.strip()) for i in p2.readlines()]
    pinn2 = p2_list[1::2]
    p2.close()

plt.clf()
plt.xlabel("Iteration")
plt.ylabel("PINN 1")
plt.plot([i for i in range(len(pinn1))], pinn1)
plt.title("PINN 1 over time ")
plt.savefig("pinn1_over_time.png")

plt.clf()
plt.xlabel("Iteration")
plt.ylabel("PINN 2")
plt.plot([i for i in range(len(pinn1))], pinn2)
plt.title("PINN 2 over time ")
plt.savefig("pinn2_over_time.png")