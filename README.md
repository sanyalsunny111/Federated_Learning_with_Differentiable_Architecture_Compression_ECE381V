# Federated_Learning_with_Differentiable_Architecture_Compression_ECE381V
Hardware heterogeneity remains a huge challenge for Federated Learning. This projects tried to solve the problem hardware heterogeneity problem in FL using Differential Architecture Compression. We have developed this class project for EE381V Advanced Computer Vision taught by Prof. Atlas Wang.

# Abstract
Despite several merits, federated learning couldn’t handle hardware heterogeneity well. Specifically, every client device has a distinct hardware configuration that differs in
storage, power, and computation capability. Hence a global model that may work for a resource-rich device may not fit for a resource-constrained device. This phenomenon limits
federated learning only to high-end resource-abundant devices. To address the challenge of hardware heterogeneity, we propose a neural architecture search-based differentiable
architecture compression (DAC) approach that computes suitable neural architectures given device configurations of the participating devices. Our experiments show that our proposed algorithm outperforms the baseline regarding the compression of a MobileNet-V2 architecture, and the DAC generated models exhibit reasonable accuracy in multiple federated learning scenarios.

![Screenshot 2022-01-29 122112](https://user-images.githubusercontent.com/36811567/151672791-bad6a3e2-ef9e-45d1-a905-ad3c3ed1bf55.png)
