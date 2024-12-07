

___

BROT 🍞 (**B**ilevel **R**outing on networks with **O**ptimal **T**ransport) is a Python implementation of the algorithms used in:

- [1] Alessandro Lonardi and Caterina De Bacco. <i>Bilevel Optimization for Traffic Mitigation in Optimal Transport Networks</i>. <a href="https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.131.267401">Phys. Rev. Lett. **131**, 267401</a> [<a href="https://arxiv.org/abs/2306.16246">arXiv</a>].

This is a scheme capable of extracting origin-destination paths on networks by making a trade off between transportation efficiency and over-trafficked links. The core algorithm alternates the integration of a system of ODEs to find passengers' shortest origin-destination routes, and Projected Stochastic Gradient Descent to mitigate traffic.

