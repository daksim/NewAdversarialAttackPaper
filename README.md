# Latest Adversarial Attack Papers
**update at 2024-05-22 15:27:35**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy**

cs.LG

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2403.01218v3) [paper-pdf](http://arxiv.org/pdf/2403.01218v3)

**Authors**: Jamie Hayes, Ilia Shumailov, Eleni Triantafillou, Amr Khalifa, Nicolas Papernot

**Abstract**: The high cost of model training makes it increasingly desirable to develop techniques for unlearning. These techniques seek to remove the influence of a training example without having to retrain the model from scratch. Intuitively, once a model has unlearned, an adversary that interacts with the model should no longer be able to tell whether the unlearned example was included in the model's training set or not. In the privacy literature, this is known as membership inference. In this work, we discuss adaptations of Membership Inference Attacks (MIAs) to the setting of unlearning (leading to their "U-MIA" counterparts). We propose a categorization of existing U-MIAs into "population U-MIAs", where the same attacker is instantiated for all examples, and "per-example U-MIAs", where a dedicated attacker is instantiated for each example. We show that the latter category, wherein the attacker tailors its membership prediction to each example under attack, is significantly stronger. Indeed, our results show that the commonly used U-MIAs in the unlearning literature overestimate the privacy protection afforded by existing unlearning techniques on both vision and language models. Our investigation reveals a large variance in the vulnerability of different examples to per-example U-MIAs. In fact, several unlearning algorithms lead to a reduced vulnerability for some, but not all, examples that we wish to unlearn, at the expense of increasing it for other examples. Notably, we find that the privacy protection for the remaining training examples may worsen as a consequence of unlearning. We also discuss the fundamental difficulty of equally protecting all examples using existing unlearning schemes, due to the different rates at which examples are unlearned. We demonstrate that naive attempts at tailoring unlearning stopping criteria to different examples fail to alleviate these issues.



## **2. Adversarial Attacks and Defenses in Automated Control Systems: A Comprehensive Benchmark**

cs.LG

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2403.13502v3) [paper-pdf](http://arxiv.org/pdf/2403.13502v3)

**Authors**: Vitaliy Pozdnyakov, Aleksandr Kovalenko, Ilya Makarov, Mikhail Drobyshevskiy, Kirill Lukyanov

**Abstract**: Integrating machine learning into Automated Control Systems (ACS) enhances decision-making in industrial process management. One of the limitations to the widespread adoption of these technologies in industry is the vulnerability of neural networks to adversarial attacks. This study explores the threats in deploying deep learning models for fault diagnosis in ACS using the Tennessee Eastman Process dataset. By evaluating three neural networks with different architectures, we subject them to six types of adversarial attacks and explore five different defense methods. Our results highlight the strong vulnerability of models to adversarial samples and the varying effectiveness of defense strategies. We also propose a novel protection approach by combining multiple defense methods and demonstrate it's efficacy. This research contributes several insights into securing machine learning within ACS, ensuring robust fault diagnosis in industrial processes.



## **3. Rethinking the Vulnerabilities of Face Recognition Systems:From a Practical Perspective**

cs.CR

19 pages

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12786v1) [paper-pdf](http://arxiv.org/pdf/2405.12786v1)

**Authors**: Jiahao Chen, Zhiqiang Shen, Yuwen Pu, Chunyi Zhou, Shouling Ji

**Abstract**: Face Recognition Systems (FRS) have increasingly integrated into critical applications, including surveillance and user authentication, highlighting their pivotal role in modern security systems. Recent studies have revealed vulnerabilities in FRS to adversarial (e.g., adversarial patch attacks) and backdoor attacks (e.g., training data poisoning), raising significant concerns about their reliability and trustworthiness. Previous studies primarily focus on traditional adversarial or backdoor attacks, overlooking the resource-intensive or privileged-manipulation nature of such threats, thus limiting their practical generalization, stealthiness, universality and robustness. Correspondingly, in this paper, we delve into the inherent vulnerabilities in FRS through user studies and preliminary explorations. By exploiting these vulnerabilities, we identify a novel attack, facial identity backdoor attack dubbed FIBA, which unveils a potentially more devastating threat against FRS:an enrollment-stage backdoor attack. FIBA circumvents the limitations of traditional attacks, enabling broad-scale disruption by allowing any attacker donning a specific trigger to bypass these systems. This implies that after a single, poisoned example is inserted into the database, the corresponding trigger becomes a universal key for any attackers to spoof the FRS. This strategy essentially challenges the conventional attacks by initiating at the enrollment stage, dramatically transforming the threat landscape by poisoning the feature database rather than the training data.



## **4. Generative AI and Large Language Models for Cyber Security: All Insights You Need**

cs.CR

50 pages, 8 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12750v1) [paper-pdf](http://arxiv.org/pdf/2405.12750v1)

**Authors**: Mohamed Amine Ferrag, Fatima Alwahedi, Ammar Battah, Bilel Cherif, Abdechakour Mechri, Norbert Tihanyi

**Abstract**: This paper provides a comprehensive review of the future of cybersecurity through Generative AI and Large Language Models (LLMs). We explore LLM applications across various domains, including hardware design security, intrusion detection, software engineering, design verification, cyber threat intelligence, malware detection, and phishing detection. We present an overview of LLM evolution and its current state, focusing on advancements in models such as GPT-4, GPT-3.5, Mixtral-8x7B, BERT, Falcon2, and LLaMA. Our analysis extends to LLM vulnerabilities, such as prompt injection, insecure output handling, data poisoning, DDoS attacks, and adversarial instructions. We delve into mitigation strategies to protect these models, providing a comprehensive look at potential attack scenarios and prevention techniques. Furthermore, we evaluate the performance of 42 LLM models in cybersecurity knowledge and hardware security, highlighting their strengths and weaknesses. We thoroughly evaluate cybersecurity datasets for LLM training and testing, covering the lifecycle from data creation to usage and identifying gaps for future research. In addition, we review new strategies for leveraging LLMs, including techniques like Half-Quadratic Quantization (HQQ), Reinforcement Learning with Human Feedback (RLHF), Direct Preference Optimization (DPO), Quantized Low-Rank Adapters (QLoRA), and Retrieval-Augmented Generation (RAG). These insights aim to enhance real-time cybersecurity defenses and improve the sophistication of LLM applications in threat detection and response. Our paper provides a foundational understanding and strategic direction for integrating LLMs into future cybersecurity frameworks, emphasizing innovation and robust model deployment to safeguard against evolving cyber threats.



## **5. A GAN-Based Data Poisoning Attack Against Federated Learning Systems and Its Countermeasure**

cs.CR

18 pages, 16 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.11440v2) [paper-pdf](http://arxiv.org/pdf/2405.11440v2)

**Authors**: Wei Sun, Bo Gao, Ke Xiong, Yuwei Wang

**Abstract**: As a distributed machine learning paradigm, federated learning (FL) is collaboratively carried out on privately owned datasets but without direct data access. Although the original intention is to allay data privacy concerns, "available but not visible" data in FL potentially brings new security threats, particularly poisoning attacks that target such "not visible" local data. Initial attempts have been made to conduct data poisoning attacks against FL systems, but cannot be fully successful due to their high chance of causing statistical anomalies. To unleash the potential for truly "invisible" attacks and build a more deterrent threat model, in this paper, a new data poisoning attack model named VagueGAN is proposed, which can generate seemingly legitimate but noisy poisoned data by untraditionally taking advantage of generative adversarial network (GAN) variants. Capable of manipulating the quality of poisoned data on demand, VagueGAN enables to trade-off attack effectiveness and stealthiness. Furthermore, a cost-effective countermeasure named Model Consistency-Based Defense (MCD) is proposed to identify GAN-poisoned data or models after finding out the consistency of GAN outputs. Extensive experiments on multiple datasets indicate that our attack method is generally much more stealthy as well as more effective in degrading FL performance with low complexity. Our defense method is also shown to be more competent in identifying GAN-poisoned data or models. The source codes are publicly available at \href{https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure}{https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure}.



## **6. How to Train a Backdoor-Robust Model on a Poisoned Dataset without Auxiliary Data?**

cs.CR

13 pages, under review

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12719v1) [paper-pdf](http://arxiv.org/pdf/2405.12719v1)

**Authors**: Yuwen Pu, Jiahao Chen, Chunyi Zhou, Zhou Feng, Qingming Li, Chunqiang Hu, Shouling Ji

**Abstract**: Backdoor attacks have attracted wide attention from academia and industry due to their great security threat to deep neural networks (DNN). Most of the existing methods propose to conduct backdoor attacks by poisoning the training dataset with different strategies, so it's critical to identify the poisoned samples and then train a clean model on the unreliable dataset in the context of defending backdoor attacks. Although numerous backdoor countermeasure researches are proposed, their inherent weaknesses render them limited in practical scenarios, such as the requirement of enough clean samples, unstable defense performance under various attack conditions, poor defense performance against adaptive attacks, and so on.Therefore, in this paper, we are committed to overcome the above limitations and propose a more practical backdoor defense method. Concretely, we first explore the inherent relationship between the potential perturbations and the backdoor trigger, and the theoretical analysis and experimental results demonstrate that the poisoned samples perform more robustness to perturbation than the clean ones. Then, based on our key explorations, we introduce AdvrBD, an Adversarial perturbation-based and robust Backdoor Defense framework, which can effectively identify the poisoned samples and train a clean model on the poisoned dataset. Constructively, our AdvrBD eliminates the requirement for any clean samples or knowledge about the poisoned dataset (e.g., poisoning ratio), which significantly improves the practicality in real-world scenarios.



## **7. Robust Classification via a Single Diffusion Model**

cs.CV

Accepted by ICML 2024

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2305.15241v2) [paper-pdf](http://arxiv.org/pdf/2305.15241v2)

**Authors**: Huanran Chen, Yinpeng Dong, Zhengyi Wang, Xiao Yang, Chengqi Duan, Hang Su, Jun Zhu

**Abstract**: Diffusion models have been applied to improve adversarial robustness of image classifiers by purifying the adversarial noises or generating realistic data for adversarial training. However, diffusion-based purification can be evaded by stronger adaptive attacks while adversarial training does not perform well under unseen threats, exhibiting inevitable limitations of these methods. To better harness the expressive power of diffusion models, this paper proposes Robust Diffusion Classifier (RDC), a generative classifier that is constructed from a pre-trained diffusion model to be adversarially robust. RDC first maximizes the data likelihood of a given input and then predicts the class probabilities of the optimized input using the conditional likelihood estimated by the diffusion model through Bayes' theorem. To further reduce the computational cost, we propose a new diffusion backbone called multi-head diffusion and develop efficient sampling strategies. As RDC does not require training on particular adversarial attacks, we demonstrate that it is more generalizable to defend against multiple unseen threats. In particular, RDC achieves $75.67\%$ robust accuracy against various $\ell_\infty$ norm-bounded adaptive attacks with $\epsilon_\infty=8/255$ on CIFAR-10, surpassing the previous state-of-the-art adversarial training models by $+4.77\%$. The results highlight the potential of generative classifiers by employing pre-trained diffusion models for adversarial robustness compared with the commonly studied discriminative classifiers. Code is available at \url{https://github.com/huanranchen/DiffusionClassifier}.



## **8. Fully Randomized Pointers**

cs.CR

24 pages, 3 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12513v1) [paper-pdf](http://arxiv.org/pdf/2405.12513v1)

**Authors**: Gregory J. Duck, Sai Dhawal Phaye, Roland H. C. Yap, Trevor E. Carlson

**Abstract**: Software security continues to be a critical concern for programs implemented in low-level programming languages such as C and C++. Many defenses have been proposed in the current literature, each with different trade-offs including performance, compatibility, and attack resistance. One general class of defense is pointer randomization or authentication, where invalid object access (e.g., memory errors) is obfuscated or denied. Many defenses rely on the program termination (e.g., crashing) to abort attacks, with the implicit assumption that an adversary cannot "brute force" the defense with multiple attack attempts. However, such assumptions do not always hold, such as hardware speculative execution attacks or network servers configured to restart on error. In such cases, we argue that most existing defenses provide only weak effective security.   In this paper, we propose Fully Randomized Pointers (FRP) as a stronger memory error defense that is resistant to even brute force attacks. The key idea is to fully randomize pointer bits -- as much as possible while also preserving binary compatibility -- rendering the relationships between pointers highly unpredictable. Furthermore, the very high degree of randomization renders brute force attacks impractical -- providing strong effective security compared to existing work. We design a new FRP encoding that is: (1) compatible with existing binary code (without recompilation); (2) decoupled from the underlying object layout; and (3) can be efficiently decoded on-the-fly to the underlying memory address. We prototype FRP in the form of a software implementation (BlueFat) to test security and compatibility, and a proof-of-concept hardware implementation (GreenFat) to evaluate performance. We show that FRP is secure, practical, and compatible at the binary level, while a hardware implementation can achieve low performance overheads (<10%).



## **9. Rethinking Robustness Assessment: Adversarial Attacks on Learning-based Quadrupedal Locomotion Controllers**

cs.RO

RSS 2024

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12424v1) [paper-pdf](http://arxiv.org/pdf/2405.12424v1)

**Authors**: Fan Shi, Chong Zhang, Takahiro Miki, Joonho Lee, Marco Hutter, Stelian Coros

**Abstract**: Legged locomotion has recently achieved remarkable success with the progress of machine learning techniques, especially deep reinforcement learning (RL). Controllers employing neural networks have demonstrated empirical and qualitative robustness against real-world uncertainties, including sensor noise and external perturbations. However, formally investigating the vulnerabilities of these locomotion controllers remains a challenge. This difficulty arises from the requirement to pinpoint vulnerabilities across a long-tailed distribution within a high-dimensional, temporally sequential space. As a first step towards quantitative verification, we propose a computational method that leverages sequential adversarial attacks to identify weaknesses in learned locomotion controllers. Our research demonstrates that, even state-of-the-art robust controllers can fail significantly under well-designed, low-magnitude adversarial sequence. Through experiments in simulation and on the real robot, we validate our approach's effectiveness, and we illustrate how the results it generates can be used to robustify the original policy and offer valuable insights into the safety of these black-box policies.



## **10. Rethinking PGD Attack: Is Sign Function Necessary?**

cs.LG

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2312.01260v2) [paper-pdf](http://arxiv.org/pdf/2312.01260v2)

**Authors**: Junjie Yang, Tianlong Chen, Xuxi Chen, Zhangyang Wang, Yingbin Liang

**Abstract**: Neural networks have demonstrated success in various domains, yet their performance can be significantly degraded by even a small input perturbation. Consequently, the construction of such perturbations, known as adversarial attacks, has gained significant attention, many of which fall within "white-box" scenarios where we have full access to the neural network. Existing attack algorithms, such as the projected gradient descent (PGD), commonly take the sign function on the raw gradient before updating adversarial inputs, thereby neglecting gradient magnitude information. In this paper, we present a theoretical analysis of how such sign-based update algorithm influences step-wise attack performance, as well as its caveat. We also interpret why previous attempts of directly using raw gradients failed. Based on that, we further propose a new raw gradient descent (RGD) algorithm that eliminates the use of sign. Specifically, we convert the constrained optimization problem into an unconstrained one, by introducing a new hidden variable of non-clipped perturbation that can move beyond the constraint. The effectiveness of the proposed RGD algorithm has been demonstrated extensively in experiments, outperforming PGD and other competitors in various settings, without incurring any additional computational overhead. The codes is available in https://github.com/JunjieYang97/RGD.



## **11. Hacking Predictors Means Hacking Cars: Using Sensitivity Analysis to Identify Trajectory Prediction Vulnerabilities for Autonomous Driving Security**

cs.CR

10 pages, 5 figures, 1 tables

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2401.10313v2) [paper-pdf](http://arxiv.org/pdf/2401.10313v2)

**Authors**: Marsalis Gibson, David Babazadeh, Claire Tomlin, Shankar Sastry

**Abstract**: Adversarial attacks on learning-based multi-modal trajectory predictors have already been demonstrated. However, there are still open questions about the effects of perturbations on inputs other than state histories, and how these attacks impact downstream planning and control. In this paper, we conduct a sensitivity analysis on two trajectory prediction models, Trajectron++ and AgentFormer. The analysis reveals that between all inputs, almost all of the perturbation sensitivities for both models lie only within the most recent position and velocity states. We additionally demonstrate that, despite dominant sensitivity on state history perturbations, an undetectable image map perturbation made with the Fast Gradient Sign Method can induce large prediction error increases in both models, revealing that these trajectory predictors are, in fact, susceptible to image-based attacks. Using an optimization-based planner and example perturbations crafted from sensitivity results, we show how these attacks can cause a vehicle to come to a sudden stop from moderate driving speeds.



## **12. Optimizing Sensor Network Design for Multiple Coverage**

cs.LG

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.09096v2) [paper-pdf](http://arxiv.org/pdf/2405.09096v2)

**Authors**: Lukas Taus, Yen-Hsi Richard Tsai

**Abstract**: Sensor placement optimization methods have been studied extensively. They can be applied to a wide range of applications, including surveillance of known environments, optimal locations for 5G towers, and placement of missile defense systems. However, few works explore the robustness and efficiency of the resulting sensor network concerning sensor failure or adversarial attacks. This paper addresses this issue by optimizing for the least number of sensors to achieve multiple coverage of non-simply connected domains by a prescribed number of sensors. We introduce a new objective function for the greedy (next-best-view) algorithm to design efficient and robust sensor networks and derive theoretical bounds on the network's optimality. We further introduce a Deep Learning model to accelerate the algorithm for near real-time computations. The Deep Learning model requires the generation of training examples. Correspondingly, we show that understanding the geometric properties of the training data set provides important insights into the performance and training process of deep learning techniques. Finally, we demonstrate that a simple parallel version of the greedy approach using a simpler objective can be highly competitive.



## **13. Efficient Model-Stealing Attacks Against Inductive Graph Neural Networks**

cs.LG

arXiv admin note: text overlap with arXiv:2112.08331 by other authors

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.12295v1) [paper-pdf](http://arxiv.org/pdf/2405.12295v1)

**Authors**: Marcin Podhajski, Jan Dubiński, Franziska Boenisch, Adam Dziedzic, Agnieszka Pregowska, Tomasz Michalak

**Abstract**: Graph Neural Networks (GNNs) are recognized as potent tools for processing real-world data organized in graph structures. Especially inductive GNNs, which enable the processing of graph-structured data without relying on predefined graph structures, are gaining importance in an increasingly wide variety of applications. As these networks demonstrate proficiency across a range of tasks, they become lucrative targets for model-stealing attacks where an adversary seeks to replicate the functionality of the targeted network. A large effort has been made to develop model-stealing attacks that focus on models trained with images and texts. However, little attention has been paid to GNNs trained on graph data. This paper introduces a novel method for unsupervised model-stealing attacks against inductive GNNs, based on graph contrasting learning and spectral graph augmentations to efficiently extract information from the target model. The proposed attack is thoroughly evaluated on six datasets. The results show that this approach demonstrates a higher level of efficiency compared to existing stealing attacks. More concretely, our attack outperforms the baseline on all benchmarks achieving higher fidelity and downstream accuracy of the stolen model while requiring fewer queries sent to the target model.



## **14. EGAN: Evolutional GAN for Ransomware Evasion**

cs.CR

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.12266v1) [paper-pdf](http://arxiv.org/pdf/2405.12266v1)

**Authors**: Daniel Commey, Benjamin Appiah, Bill K. Frimpong, Isaac Osei, Ebenezer N. A. Hammond, Garth V. Crosby

**Abstract**: Adversarial Training is a proven defense strategy against adversarial malware. However, generating adversarial malware samples for this type of training presents a challenge because the resulting adversarial malware needs to remain evasive and functional. This work proposes an attack framework, EGAN, to address this limitation. EGAN leverages an Evolution Strategy and Generative Adversarial Network to select a sequence of attack actions that can mutate a Ransomware file while preserving its original functionality. We tested this framework on popular AI-powered commercial antivirus systems listed on VirusTotal and demonstrated that our framework is capable of bypassing the majority of these systems. Moreover, we evaluated whether the EGAN attack framework can evade other commercial non-AI antivirus solutions. Our results indicate that the adversarial ransomware generated can increase the probability of evading some of them.



## **15. GAN-GRID: A Novel Generative Attack on Smart Grid Stability Prediction**

cs.CR

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.12076v1) [paper-pdf](http://arxiv.org/pdf/2405.12076v1)

**Authors**: Emad Efatinasab, Alessandro Brighente, Mirco Rampazzo, Nahal Azadi, Mauro Conti

**Abstract**: The smart grid represents a pivotal innovation in modernizing the electricity sector, offering an intelligent, digitalized energy network capable of optimizing energy delivery from source to consumer. It hence represents the backbone of the energy sector of a nation. Due to its central role, the availability of the smart grid is paramount and is hence necessary to have in-depth control of its operations and safety. To this aim, researchers developed multiple solutions to assess the smart grid's stability and guarantee that it operates in a safe state. Artificial intelligence and Machine learning algorithms have proven to be effective measures to accurately predict the smart grid's stability. Despite the presence of known adversarial attacks and potential solutions, currently, there exists no standardized measure to protect smart grids against this threat, leaving them open to new adversarial attacks. In this paper, we propose GAN-GRID a novel adversarial attack targeting the stability prediction system of a smart grid tailored to real-world constraints. Our findings reveal that an adversary armed solely with the stability model's output, devoid of data or model knowledge, can craft data classified as stable with an Attack Success Rate (ASR) of 0.99. Also by manipulating authentic data and sensor values, the attacker can amplify grid issues, potentially undetected due to a compromised stability prediction system. These results underscore the imperative of fortifying smart grid security mechanisms against adversarial manipulation to uphold system stability and reliability.



## **16. A Constraint-Enforcing Reward for Adversarial Attacks on Text Classifiers**

cs.CL

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.11904v1) [paper-pdf](http://arxiv.org/pdf/2405.11904v1)

**Authors**: Tom Roth, Inigo Jauregi Unanue, Alsharif Abuadbba, Massimo Piccardi

**Abstract**: Text classifiers are vulnerable to adversarial examples -- correctly-classified examples that are deliberately transformed to be misclassified while satisfying acceptability constraints. The conventional approach to finding adversarial examples is to define and solve a combinatorial optimisation problem over a space of allowable transformations. While effective, this approach is slow and limited by the choice of transformations. An alternate approach is to directly generate adversarial examples by fine-tuning a pre-trained language model, as is commonly done for other text-to-text tasks. This approach promises to be much quicker and more expressive, but is relatively unexplored. For this reason, in this work we train an encoder-decoder paraphrase model to generate a diverse range of adversarial examples. For training, we adopt a reinforcement learning algorithm and propose a constraint-enforcing reward that promotes the generation of valid adversarial examples. Experimental results over two text classification datasets show that our model has achieved a higher success rate than the original paraphrase model, and overall has proved more effective than other competitive attacks. Finally, we show how key design choices impact the generated examples and discuss the strengths and weaknesses of the proposed approach.



## **17. Adversarially Diversified Rehearsal Memory (ADRM): Mitigating Memory Overfitting Challenge in Continual Learning**

cs.LG

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.11829v1) [paper-pdf](http://arxiv.org/pdf/2405.11829v1)

**Authors**: Hikmat Khan, Ghulam Rasool, Nidhal Carla Bouaynaya

**Abstract**: Continual learning focuses on learning non-stationary data distribution without forgetting previous knowledge. Rehearsal-based approaches are commonly used to combat catastrophic forgetting. However, these approaches suffer from a problem called "rehearsal memory overfitting, " where the model becomes too specialized on limited memory samples and loses its ability to generalize effectively. As a result, the effectiveness of the rehearsal memory progressively decays, ultimately resulting in catastrophic forgetting of the learned tasks.   We introduce the Adversarially Diversified Rehearsal Memory (ADRM) to address the memory overfitting challenge. This novel method is designed to enrich memory sample diversity and bolster resistance against natural and adversarial noise disruptions. ADRM employs the FGSM attacks to introduce adversarially modified memory samples, achieving two primary objectives: enhancing memory diversity and fostering a robust response to continual feature drifts in memory samples.   Our contributions are as follows: Firstly, ADRM addresses overfitting in rehearsal memory by employing FGSM to diversify and increase the complexity of the memory buffer. Secondly, we demonstrate that ADRM mitigates memory overfitting and significantly improves the robustness of CL models, which is crucial for safety-critical applications. Finally, our detailed analysis of features and visualization demonstrates that ADRM mitigates feature drifts in CL memory samples, significantly reducing catastrophic forgetting and resulting in a more resilient CL model. Additionally, our in-depth t-SNE visualizations of feature distribution and the quantification of the feature similarity further enrich our understanding of feature representation in existing CL approaches. Our code is publically available at https://github.com/hikmatkhan/ADRM.



## **18. Fed-Credit: Robust Federated Learning with Credibility Management**

cs.LG

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.11758v1) [paper-pdf](http://arxiv.org/pdf/2405.11758v1)

**Authors**: Jiayan Chen, Zhirong Qian, Tianhui Meng, Xitong Gao, Tian Wang, Weijia Jia

**Abstract**: Aiming at privacy preservation, Federated Learning (FL) is an emerging machine learning approach enabling model training on decentralized devices or data sources. The learning mechanism of FL relies on aggregating parameter updates from individual clients. However, this process may pose a potential security risk due to the presence of malicious devices. Existing solutions are either costly due to the use of compute-intensive technology, or restrictive for reasons of strong assumptions such as the prior knowledge of the number of attackers and how they attack. Few methods consider both privacy constraints and uncertain attack scenarios. In this paper, we propose a robust FL approach based on the credibility management scheme, called Fed-Credit. Unlike previous studies, our approach does not require prior knowledge of the nodes and the data distribution. It maintains and employs a credibility set, which weighs the historical clients' contributions based on the similarity between the local models and global model, to adjust the global model update. The subtlety of Fed-Credit is that the time decay and attitudinal value factor are incorporated into the dynamic adjustment of the reputation weights and it boasts a computational complexity of O(n) (n is the number of the clients). We conducted extensive experiments on the MNIST and CIFAR-10 datasets under 5 types of attacks. The results exhibit superior accuracy and resilience against adversarial attacks, all while maintaining comparatively low computational complexity. Among these, on the Non-IID CIFAR-10 dataset, our algorithm exhibited performance enhancements of 19.5% and 14.5%, respectively, in comparison to the state-of-the-art algorithm when dealing with two types of data poisoning attacks.



## **19. Towards Optimal Adversarial Robust Q-learning with Bellman Infinity-error**

cs.LG

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2402.02165v2) [paper-pdf](http://arxiv.org/pdf/2402.02165v2)

**Authors**: Haoran Li, Zicheng Zhang, Wang Luo, Congying Han, Yudong Hu, Tiande Guo, Shichen Liao

**Abstract**: Establishing robust policies is essential to counter attacks or disturbances affecting deep reinforcement learning (DRL) agents. Recent studies explore state-adversarial robustness and suggest the potential lack of an optimal robust policy (ORP), posing challenges in setting strict robustness constraints. This work further investigates ORP: At first, we introduce a consistency assumption of policy (CAP) stating that optimal actions in the Markov decision process remain consistent with minor perturbations, supported by empirical and theoretical evidence. Building upon CAP, we crucially prove the existence of a deterministic and stationary ORP that aligns with the Bellman optimal policy. Furthermore, we illustrate the necessity of $L^{\infty}$-norm when minimizing Bellman error to attain ORP. This finding clarifies the vulnerability of prior DRL algorithms that target the Bellman optimal policy with $L^{1}$-norm and motivates us to train a Consistent Adversarial Robust Deep Q-Network (CAR-DQN) by minimizing a surrogate of Bellman Infinity-error. The top-tier performance of CAR-DQN across various benchmarks validates its practical effectiveness and reinforces the soundness of our theoretical analysis.



## **20. Adaptive Batch Normalization Networks for Adversarial Robustness**

cs.LG

Accepted at IEEE International Conference on Advanced Video and  Signal-based Surveillance (AVSS) 2024

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.11708v1) [paper-pdf](http://arxiv.org/pdf/2405.11708v1)

**Authors**: Shao-Yuan Lo, Vishal M. Patel

**Abstract**: Deep networks are vulnerable to adversarial examples. Adversarial Training (AT) has been a standard foundation of modern adversarial defense approaches due to its remarkable effectiveness. However, AT is extremely time-consuming, refraining it from wide deployment in practical applications. In this paper, we aim at a non-AT defense: How to design a defense method that gets rid of AT but is still robust against strong adversarial attacks? To answer this question, we resort to adaptive Batch Normalization (BN), inspired by the recent advances in test-time domain adaptation. We propose a novel defense accordingly, referred to as the Adaptive Batch Normalization Network (ABNN). ABNN employs a pre-trained substitute model to generate clean BN statistics and sends them to the target model. The target model is exclusively trained on clean data and learns to align the substitute model's BN statistics. Experimental results show that ABNN consistently improves adversarial robustness against both digital and physically realizable attacks on both image and video datasets. Furthermore, ABNN can achieve higher clean data performance and significantly lower training time complexity compared to AT-based approaches.



## **21. Geometry-Aware Instrumental Variable Regression**

cs.LG

**SubmitDate**: 2024-05-19    [abs](http://arxiv.org/abs/2405.11633v1) [paper-pdf](http://arxiv.org/pdf/2405.11633v1)

**Authors**: Heiner Kremer, Bernhard Schölkopf

**Abstract**: Instrumental variable (IV) regression can be approached through its formulation in terms of conditional moment restrictions (CMR). Building on variants of the generalized method of moments, most CMR estimators are implicitly based on approximating the population data distribution via reweightings of the empirical sample. While for large sample sizes, in the independent identically distributed (IID) setting, reweightings can provide sufficient flexibility, they might fail to capture the relevant information in presence of corrupted data or data prone to adversarial attacks. To address these shortcomings, we propose the Sinkhorn Method of Moments, an optimal transport-based IV estimator that takes into account the geometry of the data manifold through data-derivative information. We provide a simple plug-and-play implementation of our method that performs on par with related estimators in standard settings but improves robustness against data corruption and adversarial attacks.



## **22. Searching Realistic-Looking Adversarial Objects For Autonomous Driving Systems**

cs.CV

**SubmitDate**: 2024-05-19    [abs](http://arxiv.org/abs/2405.11629v1) [paper-pdf](http://arxiv.org/pdf/2405.11629v1)

**Authors**: Shengxiang Sun, Shenzhe Zhu

**Abstract**: Numerous studies on adversarial attacks targeting self-driving policies fail to incorporate realistic-looking adversarial objects, limiting real-world applicability. Building upon prior research that facilitated the transition of adversarial objects from simulations to practical applications, this paper discusses a modified gradient-based texture optimization method to discover realistic-looking adversarial objects. While retaining the core architecture and techniques of the prior research, the proposed addition involves an entity termed the 'Judge'. This agent assesses the texture of a rendered object, assigning a probability score reflecting its realism. This score is integrated into the loss function to encourage the NeRF object renderer to concurrently learn realistic and adversarial textures. The paper analyzes four strategies for developing a robust 'Judge': 1) Leveraging cutting-edge vision-language models. 2) Fine-tuning open-sourced vision-language models. 3) Pretraining neurosymbolic systems. 4) Utilizing traditional image processing techniques. Our findings indicate that strategies 1) and 4) yield less reliable outcomes, pointing towards strategies 2) or 3) as more promising directions for future research.



## **23. Struggle with Adversarial Defense? Try Diffusion**

cs.CV

**SubmitDate**: 2024-05-19    [abs](http://arxiv.org/abs/2404.08273v3) [paper-pdf](http://arxiv.org/pdf/2404.08273v3)

**Authors**: Yujie Li, Yanbin Wang, Haitao Xu, Bin Liu, Jianguo Sun, Zhenhao Guo, Wenrui Ma

**Abstract**: Adversarial attacks induce misclassification by introducing subtle perturbations. Recently, diffusion models are applied to the image classifiers to improve adversarial robustness through adversarial training or by purifying adversarial noise. However, diffusion-based adversarial training often encounters convergence challenges and high computational expenses. Additionally, diffusion-based purification inevitably causes data shift and is deemed susceptible to stronger adaptive attacks. To tackle these issues, we propose the Truth Maximization Diffusion Classifier (TMDC), a generative Bayesian classifier that builds upon pre-trained diffusion models and the Bayesian theorem. Unlike data-driven classifiers, TMDC, guided by Bayesian principles, utilizes the conditional likelihood from diffusion models to determine the class probabilities of input images, thereby insulating against the influences of data shift and the limitations of adversarial training. Moreover, to enhance TMDC's resilience against more potent adversarial attacks, we propose an optimization strategy for diffusion classifiers. This strategy involves post-training the diffusion model on perturbed datasets with ground-truth labels as conditions, guiding the diffusion model to learn the data distribution and maximizing the likelihood under the ground-truth labels. The proposed method achieves state-of-the-art performance on the CIFAR10 dataset against heavy white-box attacks and strong adaptive attacks. Specifically, TMDC achieves robust accuracies of 82.81% against $l_{\infty}$ norm-bounded perturbations and 86.05% against $l_{2}$ norm-bounded perturbations, respectively, with $\epsilon=0.05$.



## **24. On Robust Reinforcement Learning with Lipschitz-Bounded Policy Networks**

cs.LG

**SubmitDate**: 2024-05-19    [abs](http://arxiv.org/abs/2405.11432v1) [paper-pdf](http://arxiv.org/pdf/2405.11432v1)

**Authors**: Nicholas H. Barbara, Ruigang Wang, Ian R. Manchester

**Abstract**: This paper presents a study of robust policy networks in deep reinforcement learning. We investigate the benefits of policy parameterizations that naturally satisfy constraints on their Lipschitz bound, analyzing their empirical performance and robustness on two representative problems: pendulum swing-up and Atari Pong. We illustrate that policy networks with small Lipschitz bounds are significantly more robust to disturbances, random noise, and targeted adversarial attacks than unconstrained policies composed of vanilla multi-layer perceptrons or convolutional neural networks. Moreover, we find that choosing a policy parameterization with a non-conservative Lipschitz bound and an expressive, nonlinear layer architecture gives the user much finer control over the performance-robustness trade-off than existing state-of-the-art methods based on spectral normalization.



## **25. IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling Consistency**

cs.LG

Accepted to ICML 2024, 29 pages

**SubmitDate**: 2024-05-19    [abs](http://arxiv.org/abs/2405.09786v2) [paper-pdf](http://arxiv.org/pdf/2405.09786v2)

**Authors**: Linshan Hou, Ruili Feng, Zhongyun Hua, Wei Luo, Leo Yu Zhang, Yiming Li

**Abstract**: Deep neural networks (DNNs) are vulnerable to backdoor attacks, where adversaries can maliciously trigger model misclassifications by implanting a hidden backdoor during model training. This paper proposes a simple yet effective input-level backdoor detection (dubbed IBD-PSC) as a 'firewall' to filter out malicious testing images. Our method is motivated by an intriguing phenomenon, i.e., parameter-oriented scaling consistency (PSC), where the prediction confidences of poisoned samples are significantly more consistent than those of benign ones when amplifying model parameters. In particular, we provide theoretical analysis to safeguard the foundations of the PSC phenomenon. We also design an adaptive method to select BN layers to scale up for effective detection. Extensive experiments are conducted on benchmark datasets, verifying the effectiveness and efficiency of our IBD-PSC method and its resistance to adaptive attacks.



## **26. The Perception-Robustness Tradeoff in Deterministic Image Restoration**

eess.IV

**SubmitDate**: 2024-05-18    [abs](http://arxiv.org/abs/2311.09253v3) [paper-pdf](http://arxiv.org/pdf/2311.09253v3)

**Authors**: Guy Ohayon, Tomer Michaeli, Michael Elad

**Abstract**: We study the behavior of deterministic methods for solving inverse problems in imaging. These methods are commonly designed to achieve two goals: (1) attaining high perceptual quality, and (2) generating reconstructions that are consistent with the measurements. We provide a rigorous proof that the better a predictor satisfies these two requirements, the larger its Lipschitz constant must be, regardless of the nature of the degradation involved. In particular, to approach perfect perceptual quality and perfect consistency, the Lipschitz constant of the model must grow to infinity. This implies that such methods are necessarily more susceptible to adversarial attacks. We demonstrate our theory on single image super-resolution algorithms, addressing both noisy and noiseless settings. We also show how this undesired behavior can be leveraged to explore the posterior distribution, thereby allowing the deterministic model to imitate stochastic methods.



## **27. Provable Unrestricted Adversarial Training without Compromise with Generalizability**

cs.LG

**SubmitDate**: 2024-05-18    [abs](http://arxiv.org/abs/2301.09069v2) [paper-pdf](http://arxiv.org/pdf/2301.09069v2)

**Authors**: Lilin Zhang, Ning Yang, Yanchao Sun, Philip S. Yu

**Abstract**: Adversarial training (AT) is widely considered as the most promising strategy to defend against adversarial attacks and has drawn increasing interest from researchers. However, the existing AT methods still suffer from two challenges. First, they are unable to handle unrestricted adversarial examples (UAEs), which are built from scratch, as opposed to restricted adversarial examples (RAEs), which are created by adding perturbations bound by an $l_p$ norm to observed examples. Second, the existing AT methods often achieve adversarial robustness at the expense of standard generalizability (i.e., the accuracy on natural examples) because they make a tradeoff between them. To overcome these challenges, we propose a unique viewpoint that understands UAEs as imperceptibly perturbed unobserved examples. Also, we find that the tradeoff results from the separation of the distributions of adversarial examples and natural examples. Based on these ideas, we propose a novel AT approach called Provable Unrestricted Adversarial Training (PUAT), which can provide a target classifier with comprehensive adversarial robustness against both UAE and RAE, and simultaneously improve its standard generalizability. Particularly, PUAT utilizes partially labeled data to achieve effective UAE generation by accurately capturing the natural data distribution through a novel augmented triple-GAN. At the same time, PUAT extends the traditional AT by introducing the supervised loss of the target classifier into the adversarial loss and achieves the alignment between the UAE distribution, the natural data distribution, and the distribution learned by the classifier, with the collaboration of the augmented triple-GAN. Finally, the solid theoretical analysis and extensive experiments conducted on widely-used benchmarks demonstrate the superiority of PUAT.



## **28. Few-Shot API Attack Detection: Overcoming Data Scarcity with GAN-Inspired Learning**

cs.CR

8 pages, 2 figures, 7 tables

**SubmitDate**: 2024-05-18    [abs](http://arxiv.org/abs/2405.11258v1) [paper-pdf](http://arxiv.org/pdf/2405.11258v1)

**Authors**: Udi Aharon, Revital Marbel, Ran Dubin, Amit Dvir, Chen Hajaj

**Abstract**: Web applications and APIs face constant threats from malicious actors seeking to exploit vulnerabilities for illicit gains. These threats necessitate robust anomaly detection systems capable of identifying malicious API traffic efficiently despite limited and diverse datasets. This paper proposes a novel few-shot detection approach motivated by Natural Language Processing (NLP) and advanced Generative Adversarial Network (GAN)-inspired techniques. Leveraging state-of-the-art Transformer architectures, particularly RoBERTa, our method enhances the contextual understanding of API requests, leading to improved anomaly detection compared to traditional methods. We showcase the technique's versatility by demonstrating its effectiveness with both Out-of-Distribution (OOD) and Transformer-based binary classification methods on two distinct datasets: CSIC 2010 and ATRDF 2023. Our evaluations reveal consistently enhanced or, at worst, equivalent detection rates across various metrics in most vectors, highlighting the promise of our approach for improving API security.



## **29. Dynamic Quantum Key Distribution for Microgrids with Distributed Error Correction**

cs.CR

**SubmitDate**: 2024-05-18    [abs](http://arxiv.org/abs/2405.11245v1) [paper-pdf](http://arxiv.org/pdf/2405.11245v1)

**Authors**: Suman Rath, Neel Kanth Kundu, Subham Sahoo

**Abstract**: Quantum key distribution (QKD) has often been hailed as a reliable technology for secure communication in cyber-physical microgrids. Even though unauthorized key measurements are not possible in QKD, attempts to read them can disturb quantum states leading to mutations in the transmitted value. Further, inaccurate quantum keys can lead to erroneous decryption producing garbage values, destabilizing microgrid operation. QKD can also be vulnerable to node-level manipulations incorporating attack values into measurements before they are encrypted at the communication layer. To address these issues, this paper proposes a secure QKD protocol that can identify errors in keys and/or nodal measurements by observing violations in control dynamics. Additionally, the protocol uses a dynamic adjacency matrix-based formulation strategy enabling the affected nodes to reconstruct a trustworthy signal and replace it with the attacked signal in a multi-hop manner. This enables microgrids to perform nominal operations in the presence of adversaries who try to eavesdrop on the system causing an increase in the quantum bit error rate (QBER). We provide several case studies to showcase the robustness of the proposed strategy against eavesdroppers and node manipulations. The results demonstrate that it can resist unwanted observation and attack vectors that manipulate signals before encryption.



## **30. Towards Robust Policy: Enhancing Offline Reinforcement Learning with Adversarial Attacks and Defenses**

cs.LG

**SubmitDate**: 2024-05-18    [abs](http://arxiv.org/abs/2405.11206v1) [paper-pdf](http://arxiv.org/pdf/2405.11206v1)

**Authors**: Thanh Nguyen, Tung M. Luu, Tri Ton, Chang D. Yoo

**Abstract**: Offline reinforcement learning (RL) addresses the challenge of expensive and high-risk data exploration inherent in RL by pre-training policies on vast amounts of offline data, enabling direct deployment or fine-tuning in real-world environments. However, this training paradigm can compromise policy robustness, leading to degraded performance in practical conditions due to observation perturbations or intentional attacks. While adversarial attacks and defenses have been extensively studied in deep learning, their application in offline RL is limited. This paper proposes a framework to enhance the robustness of offline RL models by leveraging advanced adversarial attacks and defenses. The framework attacks the actor and critic components by perturbing observations during training and using adversarial defenses as regularization to enhance the learned policy. Four attacks and two defenses are introduced and evaluated on the D4RL benchmark. The results show the vulnerability of both the actor and critic to attacks and the effectiveness of the defenses in improving policy robustness. This framework holds promise for enhancing the reliability of offline RL models in practical scenarios.



## **31. Trustworthy Actionable Perturbations**

cs.LG

Accepted at the 41st International Conference on Machine Learning  (ICML) 2024

**SubmitDate**: 2024-05-18    [abs](http://arxiv.org/abs/2405.11195v1) [paper-pdf](http://arxiv.org/pdf/2405.11195v1)

**Authors**: Jesse Friedbaum, Sudarshan Adiga, Ravi Tandon

**Abstract**: Counterfactuals, or modified inputs that lead to a different outcome, are an important tool for understanding the logic used by machine learning classifiers and how to change an undesirable classification. Even if a counterfactual changes a classifier's decision, however, it may not affect the true underlying class probabilities, i.e. the counterfactual may act like an adversarial attack and ``fool'' the classifier. We propose a new framework for creating modified inputs that change the true underlying probabilities in a beneficial way which we call Trustworthy Actionable Perturbations (TAP). This includes a novel verification procedure to ensure that TAP change the true class probabilities instead of acting adversarially. Our framework also includes new cost, reward, and goal definitions that are better suited to effectuating change in the real world. We present PAC-learnability results for our verification procedure and theoretically analyze our new method for measuring reward. We also develop a methodology for creating TAP and compare our results to those achieved by previous counterfactual methods.



## **32. Revisiting the Robust Generalization of Adversarial Prompt Tuning**

cs.CV

**SubmitDate**: 2024-05-18    [abs](http://arxiv.org/abs/2405.11154v1) [paper-pdf](http://arxiv.org/pdf/2405.11154v1)

**Authors**: Fan Yang, Mingxuan Xia, Sangzhou Xia, Chicheng Ma, Hui Hui

**Abstract**: Understanding the vulnerability of large-scale pre-trained vision-language models like CLIP against adversarial attacks is key to ensuring zero-shot generalization capacity on various downstream tasks. State-of-the-art defense mechanisms generally adopt prompt learning strategies for adversarial fine-tuning to improve the adversarial robustness of the pre-trained model while keeping the efficiency of adapting to downstream tasks. Such a setup leads to the problem of over-fitting which impedes further improvement of the model's generalization capacity on both clean and adversarial examples. In this work, we propose an adaptive Consistency-guided Adversarial Prompt Tuning (i.e., CAPT) framework that utilizes multi-modal prompt learning to enhance the alignment of image and text features for adversarial examples and leverage the strong generalization of pre-trained CLIP to guide the model-enhancing its robust generalization on adversarial examples while maintaining its accuracy on clean ones. We also design a novel adaptive consistency objective function to balance the consistency of adversarial inputs and clean inputs between the fine-tuning model and the pre-trained model. We conduct extensive experiments across 14 datasets and 4 data sparsity schemes (from 1-shot to full training data settings) to show the superiority of CAPT over other state-of-the-art adaption methods. CAPT demonstrated excellent performance in terms of the in-distribution performance and the generalization under input distribution shift and across datasets.



## **33. Calibration Attacks: A Comprehensive Study of Adversarial Attacks on Model Confidence**

cs.LG

**SubmitDate**: 2024-05-18    [abs](http://arxiv.org/abs/2401.02718v2) [paper-pdf](http://arxiv.org/pdf/2401.02718v2)

**Authors**: Stephen Obadinma, Xiaodan Zhu, Hongyu Guo

**Abstract**: In this work, we highlight and perform a comprehensive study on calibration attacks, a form of adversarial attacks that aim to trap victim models to be heavily miscalibrated without altering their predicted labels, hence endangering the trustworthiness of the models and follow-up decision making based on their confidence. We propose four typical forms of calibration attacks: underconfidence, overconfidence, maximum miscalibration, and random confidence attacks, conducted in both the black-box and white-box setups. We demonstrate that the attacks are highly effective on both convolutional and attention-based models: with a small number of queries, they seriously skew confidence without changing the predictive performance. Given the potential danger, we further investigate the effectiveness of a wide range of adversarial defence and recalibration methods, including our proposed defences specifically designed for calibration attacks to mitigate the harm. From the ECE and KS scores, we observe that there are still significant limitations in handling calibration attacks. To the best of our knowledge, this is the first dedicated study that provides a comprehensive investigation on calibration-focused attacks. We hope this study helps attract more attention to these types of attacks and hence hamper their potential serious damages. To this end, this work also provides detailed analyses to understand the characteristics of the attacks.



## **34. Transpose Attack: Stealing Datasets with Bidirectional Training**

cs.LG

NDSS24 paper, Transpose Attack, Transposed Model. NDSS version:  https://www.ndss-symposium.org/ndss-paper/transpose-attack-stealing-datasets-with-bidirectional-training/

**SubmitDate**: 2024-05-17    [abs](http://arxiv.org/abs/2311.07389v2) [paper-pdf](http://arxiv.org/pdf/2311.07389v2)

**Authors**: Guy Amit, Mosh Levy, Yisroel Mirsky

**Abstract**: Deep neural networks are normally executed in the forward direction. However, in this work, we identify a vulnerability that enables models to be trained in both directions and on different tasks. Adversaries can exploit this capability to hide rogue models within seemingly legitimate models. In addition, in this work we show that neural networks can be taught to systematically memorize and retrieve specific samples from datasets. Together, these findings expose a novel method in which adversaries can exfiltrate datasets from protected learning environments under the guise of legitimate models. We focus on the data exfiltration attack and show that modern architectures can be used to secretly exfiltrate tens of thousands of samples with high fidelity, high enough to compromise data privacy and even train new models. Moreover, to mitigate this threat we propose a novel approach for detecting infected models.



## **35. Rethinking Graph Backdoor Attacks: A Distribution-Preserving Perspective**

cs.LG

**SubmitDate**: 2024-05-17    [abs](http://arxiv.org/abs/2405.10757v1) [paper-pdf](http://arxiv.org/pdf/2405.10757v1)

**Authors**: Zhiwei Zhang, Minhua Lin, Enyan Dai, Suhang Wang

**Abstract**: Graph Neural Networks (GNNs) have shown remarkable performance in various tasks. However, recent works reveal that GNNs are vulnerable to backdoor attacks. Generally, backdoor attack poisons the graph by attaching backdoor triggers and the target class label to a set of nodes in the training graph. A GNN trained on the poisoned graph will then be misled to predict test nodes attached with trigger to the target class. Despite their effectiveness, our empirical analysis shows that triggers generated by existing methods tend to be out-of-distribution (OOD), which significantly differ from the clean data. Hence, these injected triggers can be easily detected and pruned with widely used outlier detection methods in real-world applications. Therefore, in this paper, we study a novel problem of unnoticeable graph backdoor attacks with in-distribution (ID) triggers. To generate ID triggers, we introduce an OOD detector in conjunction with an adversarial learning strategy to generate the attributes of the triggers within distribution. To ensure a high attack success rate with ID triggers, we introduce novel modules designed to enhance trigger memorization by the victim model trained on poisoned graph. Extensive experiments on real-world datasets demonstrate the effectiveness of the proposed method in generating in distribution triggers that can by-pass various defense strategies while maintaining a high attack success rate.



## **36. Safeguarding Vision-Language Models Against Patched Visual Prompt Injectors**

cs.CV

15 pages

**SubmitDate**: 2024-05-17    [abs](http://arxiv.org/abs/2405.10529v1) [paper-pdf](http://arxiv.org/pdf/2405.10529v1)

**Authors**: Jiachen Sun, Changsheng Wang, Jiongxiao Wang, Yiwei Zhang, Chaowei Xiao

**Abstract**: Large language models have become increasingly prominent, also signaling a shift towards multimodality as the next frontier in artificial intelligence, where their embeddings are harnessed as prompts to generate textual content. Vision-language models (VLMs) stand at the forefront of this advancement, offering innovative ways to combine visual and textual data for enhanced understanding and interaction. However, this integration also enlarges the attack surface. Patch-based adversarial attack is considered the most realistic threat model in physical vision applications, as demonstrated in many existing literature. In this paper, we propose to address patched visual prompt injection, where adversaries exploit adversarial patches to generate target content in VLMs. Our investigation reveals that patched adversarial prompts exhibit sensitivity to pixel-wise randomization, a trait that remains robust even against adaptive attacks designed to counteract such defenses. Leveraging this insight, we introduce SmoothVLM, a defense mechanism rooted in smoothing techniques, specifically tailored to protect VLMs from the threat of patched visual prompt injectors. Our framework significantly lowers the attack success rate to a range between 0% and 5.0% on two leading VLMs, while achieving around 67.3% to 95.0% context recovery of the benign images, demonstrating a balance between security and usability.



## **37. ALI-DPFL: Differentially Private Federated Learning with Adaptive Local Iterations**

cs.LG

**SubmitDate**: 2024-05-17    [abs](http://arxiv.org/abs/2308.10457v8) [paper-pdf](http://arxiv.org/pdf/2308.10457v8)

**Authors**: Xinpeng Ling, Jie Fu, Kuncan Wang, Haitao Liu, Zhili Chen

**Abstract**: Federated Learning (FL) is a distributed machine learning technique that allows model training among multiple devices or organizations by sharing training parameters instead of raw data. However, adversaries can still infer individual information through inference attacks (e.g. differential attacks) on these training parameters. As a result, Differential Privacy (DP) has been widely used in FL to prevent such attacks.   We consider differentially private federated learning in a resource-constrained scenario, where both privacy budget and communication rounds are constrained. By theoretically analyzing the convergence, we can find the optimal number of local DPSGD iterations for clients between any two sequential global updates. Based on this, we design an algorithm of Differentially Private Federated Learning with Adaptive Local Iterations (ALI-DPFL). We experiment our algorithm on the MNIST, FashionMNIST and Cifar10 datasets, and demonstrate significantly better performances than previous work in the resource-constraint scenario. Code is available at https://github.com/KnightWan/ALI-DPFL.



## **38. Secure Set-Based State Estimation for Linear Systems under Adversarial Attacks on Sensors**

eess.SY

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2309.05075v2) [paper-pdf](http://arxiv.org/pdf/2309.05075v2)

**Authors**: M. Umar B. Niazi, Michelle S. Chong, Amr Alanwar, Karl H. Johansson

**Abstract**: Set-based state estimation plays a vital role in the safety verification of dynamical systems, which becomes significantly challenging when the system's sensors are susceptible to cyber-attacks. Existing methods often impose limitations on the attacker's capabilities, restricting the number of attacked sensors to be strictly less than half of the total number of sensors. This paper proposes a Secure Set-Based State Estimation (S3E) algorithm that addresses this limitation. The S3E algorithm guarantees that the true system state is contained within the estimated set, provided the initialization set encompasses the true initial state and the system is redundantly observable from the set of uncompromised sensors. The algorithm gives the estimated set as a collection of constrained zonotopes, which can be employed as robust certificates for verifying whether the system adheres to safety constraints. Furthermore, we demonstrate that the estimated set remains unaffected by attack signals of sufficiently large and also establish sufficient conditions for attack detection, identification, and filtering. This compels the attacker to inject only stealthy signals of small magnitude to evade detection, thus preserving the accuracy of the estimated set. When a few number of sensors (less than half) can be compromised, we prove that the estimated set remains bounded by a contracting set that converges to a ball whose radius is solely determined by the noise magnitude and is independent of the attack signals. To address the computational complexity of the algorithm, we offer several strategies for complexity-performance trade-offs. The efficacy of the proposed algorithm is illustrated through its application to a three-story building model.



## **39. Adversarial Robustness Guarantees for Quantum Classifiers**

quant-ph

9+12 pages, 3 figures. Comments welcome

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.10360v1) [paper-pdf](http://arxiv.org/pdf/2405.10360v1)

**Authors**: Neil Dowling, Maxwell T. West, Angus Southwell, Azar C. Nakhl, Martin Sevior, Muhammad Usman, Kavan Modi

**Abstract**: Despite their ever more widespread deployment throughout society, machine learning algorithms remain critically vulnerable to being spoofed by subtle adversarial tampering with their input data. The prospect of near-term quantum computers being capable of running {quantum machine learning} (QML) algorithms has therefore generated intense interest in their adversarial vulnerability. Here we show that quantum properties of QML algorithms can confer fundamental protections against such attacks, in certain scenarios guaranteeing robustness against classically-armed adversaries. We leverage tools from many-body physics to identify the quantum sources of this protection. Our results offer a theoretical underpinning of recent evidence which suggest quantum advantages in the search for adversarial robustness. In particular, we prove that quantum classifiers are: (i) protected against weak perturbations of data drawn from the trained distribution, (ii) protected against local attacks if they are insufficiently scrambling, and (iii) protected against universal adversarial attacks if they are sufficiently quantum chaotic. Our analytic results are supported by numerical evidence demonstrating the applicability of our theorems and the resulting robustness of a quantum classifier in practice. This line of inquiry constitutes a concrete pathway to advantage in QML, orthogonal to the usually sought improvements in model speed or accuracy.



## **40. "What do you want from theory alone?" Experimenting with Tight Auditing of Differentially Private Synthetic Data Generation**

cs.CR

To appear at Usenix Security 2024

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.10994v1) [paper-pdf](http://arxiv.org/pdf/2405.10994v1)

**Authors**: Meenatchi Sundaram Muthu Selva Annamalai, Georgi Ganev, Emiliano De Cristofaro

**Abstract**: Differentially private synthetic data generation (DP-SDG) algorithms are used to release datasets that are structurally and statistically similar to sensitive data while providing formal bounds on the information they leak. However, bugs in algorithms and implementations may cause the actual information leakage to be higher. This prompts the need to verify whether the theoretical guarantees of state-of-the-art DP-SDG implementations also hold in practice. We do so via a rigorous auditing process: we compute the information leakage via an adversary playing a distinguishing game and running membership inference attacks (MIAs). If the leakage observed empirically is higher than the theoretical bounds, we identify a DP violation; if it is non-negligibly lower, the audit is loose.   We audit six DP-SDG implementations using different datasets and threat models and find that black-box MIAs commonly used against DP-SDGs are severely limited in power, yielding remarkably loose empirical privacy estimates. We then consider MIAs in stronger threat models, i.e., passive and active white-box, using both existing and newly proposed attacks. Overall, we find that, currently, we do not only need white-box MIAs but also worst-case datasets to tightly estimate the privacy leakage from DP-SDGs. Finally, we show that our automated auditing procedure finds both known DP violations (in 4 out of the 6 implementations) as well as a new one in the DPWGAN implementation that was successfully submitted to the NIST DP Synthetic Data Challenge.   The source code needed to reproduce our experiments is available from https://github.com/spalabucr/synth-audit.



## **41. Protecting Your LLMs with Information Bottleneck**

cs.CL

23 pages, 7 figures, 8 tables

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2404.13968v2) [paper-pdf](http://arxiv.org/pdf/2404.13968v2)

**Authors**: Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.



## **42. Adversarial Robustness for Visual Grounding of Multimodal Large Language Models**

cs.CV

ICLR 2024 Workshop on Reliable and Responsible Foundation Models

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09981v1) [paper-pdf](http://arxiv.org/pdf/2405.09981v1)

**Authors**: Kuofeng Gao, Yang Bai, Jiawang Bai, Yong Yang, Shu-Tao Xia

**Abstract**: Multi-modal Large Language Models (MLLMs) have recently achieved enhanced performance across various vision-language tasks including visual grounding capabilities. However, the adversarial robustness of visual grounding remains unexplored in MLLMs. To fill this gap, we use referring expression comprehension (REC) as an example task in visual grounding and propose three adversarial attack paradigms as follows. Firstly, untargeted adversarial attacks induce MLLMs to generate incorrect bounding boxes for each object. Besides, exclusive targeted adversarial attacks cause all generated outputs to the same target bounding box. In addition, permuted targeted adversarial attacks aim to permute all bounding boxes among different objects within a single image. Extensive experiments demonstrate that the proposed methods can successfully attack visual grounding capabilities of MLLMs. Our methods not only provide a new perspective for designing novel attacks but also serve as a strong baseline for improving the adversarial robustness for visual grounding of MLLMs.



## **43. Deepfake Generation and Detection: A Benchmark and Survey**

cs.CV

We closely follow the latest developments in  https://github.com/flyingby/Awesome-Deepfake-Generation-and-Detection

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2403.17881v4) [paper-pdf](http://arxiv.org/pdf/2403.17881v4)

**Authors**: Gan Pei, Jiangning Zhang, Menghan Hu, Zhenyu Zhang, Chengjie Wang, Yunsheng Wu, Guangtao Zhai, Jian Yang, Chunhua Shen, Dacheng Tao

**Abstract**: Deepfake is a technology dedicated to creating highly realistic facial images and videos under specific conditions, which has significant application potential in fields such as entertainment, movie production, digital human creation, to name a few. With the advancements in deep learning, techniques primarily represented by Variational Autoencoders and Generative Adversarial Networks have achieved impressive generation results. More recently, the emergence of diffusion models with powerful generation capabilities has sparked a renewed wave of research. In addition to deepfake generation, corresponding detection technologies continuously evolve to regulate the potential misuse of deepfakes, such as for privacy invasion and phishing attacks. This survey comprehensively reviews the latest developments in deepfake generation and detection, summarizing and analyzing current state-of-the-arts in this rapidly evolving field. We first unify task definitions, comprehensively introduce datasets and metrics, and discuss developing technologies. Then, we discuss the development of several related sub-fields and focus on researching four representative deepfake fields: face swapping, face reenactment, talking face generation, and facial attribute editing, as well as forgery detection. Subsequently, we comprehensively benchmark representative methods on popular datasets for each field, fully evaluating the latest and influential published works. Finally, we analyze challenges and future research directions of the discussed fields.



## **44. Infrared Adversarial Car Stickers**

cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09924v1) [paper-pdf](http://arxiv.org/pdf/2405.09924v1)

**Authors**: Xiaopei Zhu, Yuqiu Liu, Zhanhao Hu, Jianmin Li, Xiaolin Hu

**Abstract**: Infrared physical adversarial examples are of great significance for studying the security of infrared AI systems that are widely used in our lives such as autonomous driving. Previous infrared physical attacks mainly focused on 2D infrared pedestrian detection which may not fully manifest its destructiveness to AI systems. In this work, we propose a physical attack method against infrared detectors based on 3D modeling, which is applied to a real car. The goal is to design a set of infrared adversarial stickers to make cars invisible to infrared detectors at various viewing angles, distances, and scenes. We build a 3D infrared car model with real infrared characteristics and propose an infrared adversarial pattern generation method based on 3D mesh shadow. We propose a 3D control points-based mesh smoothing algorithm and use a set of smoothness loss functions to enhance the smoothness of adversarial meshes and facilitate the sticker implementation. Besides, We designed the aluminum stickers and conducted physical experiments on two real Mercedes-Benz A200L cars. Our adversarial stickers hid the cars from Faster RCNN, an object detector, at various viewing angles, distances, and scenes. The attack success rate (ASR) was 91.49% for real cars. In comparison, the ASRs of random stickers and no sticker were only 6.21% and 0.66%, respectively. In addition, the ASRs of the designed stickers against six unseen object detectors such as YOLOv3 and Deformable DETR were between 73.35%-95.80%, showing good transferability of the attack performance across detectors.



## **45. DiffAM: Diffusion-based Adversarial Makeup Transfer for Facial Privacy Protection**

cs.CV

16 pages, 11 figures

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09882v1) [paper-pdf](http://arxiv.org/pdf/2405.09882v1)

**Authors**: Yuhao Sun, Lingyun Yu, Hongtao Xie, Jiaming Li, Yongdong Zhang

**Abstract**: With the rapid development of face recognition (FR) systems, the privacy of face images on social media is facing severe challenges due to the abuse of unauthorized FR systems. Some studies utilize adversarial attack techniques to defend against malicious FR systems by generating adversarial examples. However, the generated adversarial examples, i.e., the protected face images, tend to suffer from subpar visual quality and low transferability. In this paper, we propose a novel face protection approach, dubbed DiffAM, which leverages the powerful generative ability of diffusion models to generate high-quality protected face images with adversarial makeup transferred from reference images. To be specific, we first introduce a makeup removal module to generate non-makeup images utilizing a fine-tuned diffusion model with guidance of textual prompts in CLIP space. As the inverse process of makeup transfer, makeup removal can make it easier to establish the deterministic relationship between makeup domain and non-makeup domain regardless of elaborate text prompts. Then, with this relationship, a CLIP-based makeup loss along with an ensemble attack strategy is introduced to jointly guide the direction of adversarial makeup domain, achieving the generation of protected face images with natural-looking makeup and high black-box transferability. Extensive experiments demonstrate that DiffAM achieves higher visual quality and attack success rates with a gain of 12.98% under black-box setting compared with the state of the arts. The code will be available at https://github.com/HansSunY/DiffAM.



## **46. Box-Free Model Watermarks Are Prone to Black-Box Removal Attacks**

cs.CV

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09863v1) [paper-pdf](http://arxiv.org/pdf/2405.09863v1)

**Authors**: Haonan An, Guang Hua, Zhiping Lin, Yuguang Fang

**Abstract**: Box-free model watermarking is an emerging technique to safeguard the intellectual property of deep learning models, particularly those for low-level image processing tasks. Existing works have verified and improved its effectiveness in several aspects. However, in this paper, we reveal that box-free model watermarking is prone to removal attacks, even under the real-world threat model such that the protected model and the watermark extractor are in black boxes. Under this setting, we carry out three studies. 1) We develop an extractor-gradient-guided (EGG) remover and show its effectiveness when the extractor uses ReLU activation only. 2) More generally, for an unknown extractor, we leverage adversarial attacks and design the EGG remover based on the estimated gradients. 3) Under the most stringent condition that the extractor is inaccessible, we design a transferable remover based on a set of private proxy models. In all cases, the proposed removers can successfully remove embedded watermarks while preserving the quality of the processed images, and we also demonstrate that the EGG remover can even replace the watermarks. Extensive experimental results verify the effectiveness and generalizability of the proposed attacks, revealing the vulnerabilities of the existing box-free methods and calling for further research.



## **47. Manifold Integrated Gradients: Riemannian Geometry for Feature Attribution**

cs.LG

Accepted at ICML 2024

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09800v1) [paper-pdf](http://arxiv.org/pdf/2405.09800v1)

**Authors**: Eslam Zaher, Maciej Trzaskowski, Quan Nguyen, Fred Roosta

**Abstract**: In this paper, we dive into the reliability concerns of Integrated Gradients (IG), a prevalent feature attribution method for black-box deep learning models. We particularly address two predominant challenges associated with IG: the generation of noisy feature visualizations for vision models and the vulnerability to adversarial attributional attacks. Our approach involves an adaptation of path-based feature attribution, aligning the path of attribution more closely to the intrinsic geometry of the data manifold. Our experiments utilise deep generative models applied to several real-world image datasets. They demonstrate that IG along the geodesics conforms to the curved geometry of the Riemannian data manifold, generating more perceptually intuitive explanations and, subsequently, substantially increasing robustness to targeted attributional attacks.



## **48. Benchmark Early and Red Team Often: A Framework for Assessing and Managing Dual-Use Hazards of AI Foundation Models**

cs.CR

62 pages

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.10986v1) [paper-pdf](http://arxiv.org/pdf/2405.10986v1)

**Authors**: Anthony M. Barrett, Krystal Jackson, Evan R. Murphy, Nada Madkour, Jessica Newman

**Abstract**: A concern about cutting-edge or "frontier" AI foundation models is that an adversary may use the models for preparing chemical, biological, radiological, nuclear, (CBRN), cyber, or other attacks. At least two methods can identify foundation models with potential dual-use capability; each has advantages and disadvantages: A. Open benchmarks (based on openly available questions and answers), which are low-cost but accuracy-limited by the need to omit security-sensitive details; and B. Closed red team evaluations (based on private evaluation by CBRN and cyber experts), which are higher-cost but can achieve higher accuracy by incorporating sensitive details. We propose a research and risk-management approach using a combination of methods including both open benchmarks and closed red team evaluations, in a way that leverages advantages of both methods. We recommend that one or more groups of researchers with sufficient resources and access to a range of near-frontier and frontier foundation models run a set of foundation models through dual-use capability evaluation benchmarks and red team evaluations, then analyze the resulting sets of models' scores on benchmark and red team evaluations to see how correlated those are. If, as we expect, there is substantial correlation between the dual-use potential benchmark scores and the red team evaluation scores, then implications include the following: The open benchmarks should be used frequently during foundation model development as a quick, low-cost measure of a model's dual-use potential; and if a particular model gets a high score on the dual-use potential benchmark, then more in-depth red team assessments of that model's dual-use capability should be performed. We also discuss limitations and mitigations for our approach, e.g., if model developers try to game benchmarks by including a version of benchmark test data in a model's training data.



## **49. Towards Evaluating the Robustness of Automatic Speech Recognition Systems via Audio Style Transfer**

cs.SD

Accepted to SecTL (AsiaCCS Workshop) 2024

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09470v1) [paper-pdf](http://arxiv.org/pdf/2405.09470v1)

**Authors**: Weifei Jin, Yuxin Cao, Junjie Su, Qi Shen, Kai Ye, Derui Wang, Jie Hao, Ziyao Liu

**Abstract**: In light of the widespread application of Automatic Speech Recognition (ASR) systems, their security concerns have received much more attention than ever before, primarily due to the susceptibility of Deep Neural Networks. Previous studies have illustrated that surreptitiously crafting adversarial perturbations enables the manipulation of speech recognition systems, resulting in the production of malicious commands. These attack methods mostly require adding noise perturbations under $\ell_p$ norm constraints, inevitably leaving behind artifacts of manual modifications. Recent research has alleviated this limitation by manipulating style vectors to synthesize adversarial examples based on Text-to-Speech (TTS) synthesis audio. However, style modifications based on optimization objectives significantly reduce the controllability and editability of audio styles. In this paper, we propose an attack on ASR systems based on user-customized style transfer. We first test the effect of Style Transfer Attack (STA) which combines style transfer and adversarial attack in sequential order. And then, as an improvement, we propose an iterative Style Code Attack (SCA) to maintain audio quality. Experimental results show that our method can meet the need for user-customized styles and achieve a success rate of 82% in attacks, while keeping sound naturalness due to our user study.



## **50. Properties that allow or prohibit transferability of adversarial attacks among quantized networks**

cs.LG

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09598v1) [paper-pdf](http://arxiv.org/pdf/2405.09598v1)

**Authors**: Abhishek Shrestha, Jürgen Großmann

**Abstract**: Deep Neural Networks (DNNs) are known to be vulnerable to adversarial examples. Further, these adversarial examples are found to be transferable from the source network in which they are crafted to a black-box target network. As the trend of using deep learning on embedded devices grows, it becomes relevant to study the transferability properties of adversarial examples among compressed networks. In this paper, we consider quantization as a network compression technique and evaluate the performance of transfer-based attacks when the source and target networks are quantized at different bitwidths. We explore how algorithm specific properties affect transferability by considering various adversarial example generation algorithms. Furthermore, we examine transferability in a more realistic scenario where the source and target networks may differ in bitwidth and other model-related properties like capacity and architecture. We find that although quantization reduces transferability, certain attack types demonstrate an ability to enhance it. Additionally, the average transferability of adversarial examples among quantized versions of a network can be used to estimate the transferability to quantized target networks with varying capacity and architecture.



