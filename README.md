# Latest Adversarial Attack Papers
**update at 2024-02-18 11:12:29**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. A Trembling House of Cards? Mapping Adversarial Attacks against Language Agents**

cs.CL

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.10196v1) [paper-pdf](http://arxiv.org/pdf/2402.10196v1)

**Authors**: Lingbo Mo, Zeyi Liao, Boyuan Zheng, Yu Su, Chaowei Xiao, Huan Sun

**Abstract**: Language agents powered by large language models (LLMs) have seen exploding development. Their capability of using language as a vehicle for thought and communication lends an incredible level of flexibility and versatility. People have quickly capitalized on this capability to connect LLMs to a wide range of external components and environments: databases, tools, the Internet, robotic embodiment, etc. Many believe an unprecedentedly powerful automation technology is emerging. However, new automation technologies come with new safety risks, especially for intricate systems like language agents. There is a surprisingly large gap between the speed and scale of their development and deployment and our understanding of their safety risks. Are we building a house of cards? In this position paper, we present the first systematic effort in mapping adversarial attacks against language agents. We first present a unified conceptual framework for agents with three major components: Perception, Brain, and Action. Under this framework, we present a comprehensive discussion and propose 12 potential attack scenarios against different components of an agent, covering different attack strategies (e.g., input manipulation, adversarial demonstrations, jailbreaking, backdoors). We also draw connections to successful attack strategies previously applied to LLMs. We emphasize the urgency to gain a thorough understanding of language agent risks before their widespread deployment.



## **2. Transaction Capacity, Security and Latency in Blockchains**

cs.CR

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.10138v1) [paper-pdf](http://arxiv.org/pdf/2402.10138v1)

**Authors**: Mustafa Doger, Sennur Ulukus

**Abstract**: We analyze how secure a block is after the block becomes k-deep, i.e., security-latency, for Nakamoto consensus under an exponential network delay model. We give parameter regimes for which transactions are safe when sufficiently deep in the chain. We compare our results for Nakamoto consensus under bounded network delay models and obtain analogous bounds for safety violation threshold. Next, modeling the blockchain system as a batch service queue with exponential network delay, we connect the security-latency analysis to sustainable transaction rate of the queue system. As our model assumes exponential network delay, batch service queue models give a meaningful trade-off between transaction capacity, security and latency. As adversary can attack the queue service to hamper the service process, we consider two different attacks for adversary. In an extreme scenario, we modify the selfish-mining attack for this purpose and consider its effect on the sustainable transaction rate of the queue.



## **3. Indiscriminate Data Poisoning Attacks on Neural Networks**

cs.LG

Accepted to TMLR in 2022

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2204.09092v2) [paper-pdf](http://arxiv.org/pdf/2204.09092v2)

**Authors**: Yiwei Lu, Gautam Kamath, Yaoliang Yu

**Abstract**: Data poisoning attacks, in which a malicious adversary aims to influence a model by injecting "poisoned" data into the training process, have attracted significant recent attention. In this work, we take a closer look at existing poisoning attacks and connect them with old and new algorithms for solving sequential Stackelberg games. By choosing an appropriate loss function for the attacker and optimizing with algorithms that exploit second-order information, we design poisoning attacks that are effective on neural networks. We present efficient implementations that exploit modern auto-differentiation packages and allow simultaneous and coordinated generation of tens of thousands of poisoned points, in contrast to existing methods that generate poisoned points one by one. We further perform extensive experiments that empirically explore the effect of data poisoning attacks on deep neural networks.



## **4. How Much Does Each Datapoint Leak Your Privacy? Quantifying the Per-datum Membership Leakage**

cs.LG

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.10065v1) [paper-pdf](http://arxiv.org/pdf/2402.10065v1)

**Authors**: Achraf Azize, Debabrota Basu

**Abstract**: We study the per-datum Membership Inference Attacks (MIAs), where an attacker aims to infer whether a fixed target datum has been included in the input dataset of an algorithm and thus, violates privacy. First, we define the membership leakage of a datum as the advantage of the optimal adversary targeting to identify it. Then, we quantify the per-datum membership leakage for the empirical mean, and show that it depends on the Mahalanobis distance between the target datum and the data-generating distribution. We further assess the effect of two privacy defences, i.e. adding Gaussian noise and sub-sampling. We quantify exactly how both of them decrease the per-datum membership leakage. Our analysis builds on a novel proof technique that combines an Edgeworth expansion of the likelihood ratio test and a Lindeberg-Feller central limit theorem. Our analysis connects the existing likelihood ratio and scalar product attacks, and also justifies different canary selection strategies used in the privacy auditing literature. Finally, our experiments demonstrate the impacts of the leakage score, the sub-sampling ratio and the noise scale on the per-datum membership leakage as indicated by the theory.



## **5. Protect Your Score: Contact Tracing With Differential Privacy Guarantees**

cs.CR

Accepted to The 38th Annual AAAI Conference on Artificial  Intelligence (AAAI 2024)

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2312.11581v2) [paper-pdf](http://arxiv.org/pdf/2312.11581v2)

**Authors**: Rob Romijnders, Christos Louizos, Yuki M. Asano, Max Welling

**Abstract**: The pandemic in 2020 and 2021 had enormous economic and societal consequences, and studies show that contact tracing algorithms can be key in the early containment of the virus. While large strides have been made towards more effective contact tracing algorithms, we argue that privacy concerns currently hold deployment back. The essence of a contact tracing algorithm constitutes the communication of a risk score. Yet, it is precisely the communication and release of this score to a user that an adversary can leverage to gauge the private health status of an individual. We pinpoint a realistic attack scenario and propose a contact tracing algorithm with differential privacy guarantees against this attack. The algorithm is tested on the two most widely used agent-based COVID19 simulators and demonstrates superior performance in a wide range of settings. Especially for realistic test scenarios and while releasing each risk score with epsilon=1 differential privacy, we achieve a two to ten-fold reduction in the infection rate of the virus. To the best of our knowledge, this presents the first contact tracing algorithm with differential privacy guarantees when revealing risk scores for COVID19.



## **6. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

cs.SD

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2305.17000v2) [paper-pdf](http://arxiv.org/pdf/2305.17000v2)

**Authors**: Matías P. Pizarro B., Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic for distinguishing target adversarial examples against clean and noisy data of 99\% and 97\%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.



## **7. Adversarial Quantum Machine Learning: An Information-Theoretic Generalization Analysis**

quant-ph

10 pages, 2 figures. Fixed a typo (wrong inequality sign) in lemma 2  and extended to cover the whole range of values of p. Added reference on  inequalities in trace norms

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.00176v2) [paper-pdf](http://arxiv.org/pdf/2402.00176v2)

**Authors**: Petros Georgiou, Sharu Theresa Jose, Osvaldo Simeone

**Abstract**: In a manner analogous to their classical counterparts, quantum classifiers are vulnerable to adversarial attacks that perturb their inputs. A promising countermeasure is to train the quantum classifier by adopting an attack-aware, or adversarial, loss function. This paper studies the generalization properties of quantum classifiers that are adversarially trained against bounded-norm white-box attacks. Specifically, a quantum adversary maximizes the classifier's loss by transforming an input state $\rho(x)$ into a state $\lambda$ that is $\epsilon$-close to the original state $\rho(x)$ in $p$-Schatten distance. Under suitable assumptions on the quantum embedding $\rho(x)$, we derive novel information-theoretic upper bounds on the generalization error of adversarially trained quantum classifiers for $p = 1$ and $p = \infty$. The derived upper bounds consist of two terms: the first is an exponential function of the 2-R\'enyi mutual information between classical data and quantum embedding, while the second term scales linearly with the adversarial perturbation size $\epsilon$. Both terms are shown to decrease as $1/\sqrt{T}$ over the training set size $T$ . An extension is also considered in which the adversary assumed during training has different parameters $p$ and $\epsilon$ as compared to the adversary affecting the test inputs. Finally, we validate our theoretical findings with numerical experiments for a synthetic setting.



## **8. Camouflage is all you need: Evaluating and Enhancing Language Model Robustness Against Camouflage Adversarial Attacks**

cs.CL

19 pages, 8 figures, 5 tables

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.09874v1) [paper-pdf](http://arxiv.org/pdf/2402.09874v1)

**Authors**: Álvaro Huertas-García, Alejandro Martín, Javier Huertas-Tato, David Camacho

**Abstract**: Adversarial attacks represent a substantial challenge in Natural Language Processing (NLP). This study undertakes a systematic exploration of this challenge in two distinct phases: vulnerability evaluation and resilience enhancement of Transformer-based models under adversarial attacks.   In the evaluation phase, we assess the susceptibility of three Transformer configurations, encoder-decoder, encoder-only, and decoder-only setups, to adversarial attacks of escalating complexity across datasets containing offensive language and misinformation. Encoder-only models manifest a 14% and 21% performance drop in offensive language detection and misinformation detection tasks, respectively. Decoder-only models register a 16% decrease in both tasks, while encoder-decoder models exhibit a maximum performance drop of 14% and 26% in the respective tasks.   The resilience-enhancement phase employs adversarial training, integrating pre-camouflaged and dynamically altered data. This approach effectively reduces the performance drop in encoder-only models to an average of 5% in offensive language detection and 2% in misinformation detection tasks. Decoder-only models, occasionally exceeding original performance, limit the performance drop to 7% and 2% in the respective tasks. Although not surpassing the original performance, Encoder-decoder models can reduce the drop to an average of 6% and 2% respectively.   Results suggest a trade-off between performance and robustness, with some models maintaining similar performance while gaining robustness. Our study and adversarial training techniques have been incorporated into an open-source tool for generating camouflaged datasets. However, methodology effectiveness depends on the specific camouflage technique and data encountered, emphasizing the need for continued exploration.



## **9. Fooling the Image Dehazing Models by First Order Gradient**

cs.CV

This paper is accepted by IEEE Transactions on Circuits and Systems  for Video Technology (TCSVT)

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2303.17255v2) [paper-pdf](http://arxiv.org/pdf/2303.17255v2)

**Authors**: Jie Gui, Xiaofeng Cong, Chengwei Peng, Yuan Yan Tang, James Tin-Yau Kwok

**Abstract**: The research on the single image dehazing task has been widely explored. However, as far as we know, no comprehensive study has been conducted on the robustness of the well-trained dehazing models. Therefore, there is no evidence that the dehazing networks can resist malicious attacks. In this paper, we focus on designing a group of attack methods based on first order gradient to verify the robustness of the existing dehazing algorithms. By analyzing the general purpose of image dehazing task, four attack methods are proposed, which are predicted dehazed image attack, hazy layer mask attack, haze-free image attack and haze-preserved attack. The corresponding experiments are conducted on six datasets with different scales. Further, the defense strategy based on adversarial training is adopted for reducing the negative effects caused by malicious attacks. In summary, this paper defines a new challenging problem for the image dehazing area, which can be called as adversarial attack on dehazing networks (AADN). Code and Supplementary Material are available at https://github.com/Xiaofeng-life/AADN Dehazing.



## **10. Exploring the Adversarial Capabilities of Large Language Models**

cs.AI

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.09132v2) [paper-pdf](http://arxiv.org/pdf/2402.09132v2)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.



## **11. Exploiting Alpha Transparency In Language And Vision-Based AI Systems**

cs.CV

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.09671v1) [paper-pdf](http://arxiv.org/pdf/2402.09671v1)

**Authors**: David Noever, Forrest McKee

**Abstract**: This investigation reveals a novel exploit derived from PNG image file formats, specifically their alpha transparency layer, and its potential to fool multiple AI vision systems. Our method uses this alpha layer as a clandestine channel invisible to human observers but fully actionable by AI image processors. The scope tested for the vulnerability spans representative vision systems from Apple, Microsoft, Google, Salesforce, Nvidia, and Facebook, highlighting the attack's potential breadth. This vulnerability challenges the security protocols of existing and fielded vision systems, from medical imaging to autonomous driving technologies. Our experiments demonstrate that the affected systems, which rely on convolutional neural networks or the latest multimodal language models, cannot quickly mitigate these vulnerabilities through simple patches or updates. Instead, they require retraining and architectural changes, indicating a persistent hole in multimodal technologies without some future adversarial hardening against such vision-language exploits.



## **12. Break it, Imitate it, Fix it: Robustness by Generating Human-Like Attacks**

cs.LG

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2310.16955v2) [paper-pdf](http://arxiv.org/pdf/2310.16955v2)

**Authors**: Aradhana Sinha, Ananth Balashankar, Ahmad Beirami, Thi Avrahami, Jilin Chen, Alex Beutel

**Abstract**: Real-world natural language processing systems need to be robust to human adversaries. Collecting examples of human adversaries for training is an effective but expensive solution. On the other hand, training on synthetic attacks with small perturbations - such as word-substitution - does not actually improve robustness to human adversaries. In this paper, we propose an adversarial training framework that uses limited human adversarial examples to generate more useful adversarial examples at scale. We demonstrate the advantages of this system on the ANLI and hate speech detection benchmark datasets - both collected via an iterative, adversarial human-and-model-in-the-loop procedure. Compared to training only on observed human attacks, also training on our synthetic adversarial examples improves model robustness to future rounds. In ANLI, we see accuracy gains on the current set of attacks (44.1%$\,\to\,$50.1%) and on two future unseen rounds of human generated attacks (32.5%$\,\to\,$43.4%, and 29.4%$\,\to\,$40.2%). In hate speech detection, we see AUC gains on current attacks (0.76 $\to$ 0.84) and a future round (0.77 $\to$ 0.79). Attacks from methods that do not learn the distribution of existing human adversaries, meanwhile, degrade robustness.



## **13. How Secure Are Large Language Models (LLMs) for Navigation in Urban Environments?**

cs.RO

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09546v1) [paper-pdf](http://arxiv.org/pdf/2402.09546v1)

**Authors**: Congcong Wen, Jiazhao Liang, Shuaihang Yuan, Hao Huang, Yi Fang

**Abstract**: In the field of robotics and automation, navigation systems based on Large Language Models (LLMs) have recently shown impressive performance. However, the security aspects of these systems have received relatively less attention. This paper pioneers the exploration of vulnerabilities in LLM-based navigation models in urban outdoor environments, a critical area given the technology's widespread application in autonomous driving, logistics, and emergency services. Specifically, we introduce a novel Navigational Prompt Suffix (NPS) Attack that manipulates LLM-based navigation models by appending gradient-derived suffixes to the original navigational prompt, leading to incorrect actions. We conducted comprehensive experiments on an LLMs-based navigation model that employs various LLMs for reasoning. Our results, derived from the Touchdown and Map2Seq street-view datasets under both few-shot learning and fine-tuning configurations, demonstrate notable performance declines across three metrics in the face of both white-box and black-box attacks. These results highlight the generalizability and transferability of the NPS Attack, emphasizing the need for enhanced security in LLM-based navigation systems. As an initial countermeasure, we propose the Navigational Prompt Engineering (NPE) Defense strategy, concentrating on navigation-relevant keywords to reduce the impact of adversarial suffixes. While initial findings indicate that this strategy enhances navigational safety, there remains a critical need for the wider research community to develop stronger defense methods to effectively tackle the real-world challenges faced by these systems.



## **14. NeuroBench: An Open-Source Benchmark Framework for the Standardization of Methodology in Brainwave-based Authentication Research**

cs.CR

21 pages, 5 Figures, 3 tables, Submitted to the Journal of  Information Security and Applications

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.08656v2) [paper-pdf](http://arxiv.org/pdf/2402.08656v2)

**Authors**: Avinash Kumar Chaurasia, Matin Fallahi, Thorsten Strufe, Philipp Terhörst, Patricia Arias Cabarcos

**Abstract**: Biometric systems based on brain activity have been proposed as an alternative to passwords or to complement current authentication techniques. By leveraging the unique brainwave patterns of individuals, these systems offer the possibility of creating authentication solutions that are resistant to theft, hands-free, accessible, and potentially even revocable. However, despite the growing stream of research in this area, faster advance is hindered by reproducibility problems. Issues such as the lack of standard reporting schemes for performance results and system configuration, or the absence of common evaluation benchmarks, make comparability and proper assessment of different biometric solutions challenging. Further, barriers are erected to future work when, as so often, source code is not published open access. To bridge this gap, we introduce NeuroBench, a flexible open source tool to benchmark brainwave-based authentication models. It incorporates nine diverse datasets, implements a comprehensive set of pre-processing parameters and machine learning algorithms, enables testing under two common adversary models (known vs unknown attacker), and allows researchers to generate full performance reports and visualizations. We use NeuroBench to investigate the shallow classifiers and deep learning-based approaches proposed in the literature, and to test robustness across multiple sessions. We observe a 37.6% reduction in Equal Error Rate (EER) for unknown attacker scenarios (typically not tested in the literature), and we highlight the importance of session variability to brainwave authentication. All in all, our results demonstrate the viability and relevance of NeuroBench in streamlining fair comparisons of algorithms, thereby furthering the advancement of brainwave-based authentication through robust methodological practices.



## **15. Asymmetric Bias in Text-to-Image Generation with Adversarial Attacks**

cs.LG

preprint version

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2312.14440v2) [paper-pdf](http://arxiv.org/pdf/2312.14440v2)

**Authors**: Haz Sameen Shahgir, Xianghao Kong, Greg Ver Steeg, Yue Dong

**Abstract**: The widespread use of Text-to-Image (T2I) models in content generation requires careful examination of their safety, including their robustness to adversarial attacks. Despite extensive research on adversarial attacks, the reasons for their effectiveness remain underexplored. This paper presents an empirical study on adversarial attacks against T2I models, focusing on analyzing factors associated with attack success rates (ASR). We introduce a new attack objective - entity swapping using adversarial suffixes and two gradient-based attack algorithms. Human and automatic evaluations reveal the asymmetric nature of ASRs on entity swap: for example, it is easier to replace "human" with "robot" in the prompt "a human dancing in the rain." with an adversarial suffix, but the reverse replacement is significantly harder. We further propose probing metrics to establish indicative signals from the model's beliefs to the adversarial ASR. We identify conditions that result in a success probability of 60% for adversarial attacks and others where this likelihood drops below 5%.



## **16. Only My Model On My Data: A Privacy Preserving Approach Protecting one Model and Deceiving Unauthorized Black-Box Models**

cs.CV

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09316v1) [paper-pdf](http://arxiv.org/pdf/2402.09316v1)

**Authors**: Weiheng Chai, Brian Testa, Huantao Ren, Asif Salekin, Senem Velipasalar

**Abstract**: Deep neural networks are extensively applied to real-world tasks, such as face recognition and medical image classification, where privacy and data protection are critical. Image data, if not protected, can be exploited to infer personal or contextual information. Existing privacy preservation methods, like encryption, generate perturbed images that are unrecognizable to even humans. Adversarial attack approaches prohibit automated inference even for authorized stakeholders, limiting practical incentives for commercial and widespread adaptation. This pioneering study tackles an unexplored practical privacy preservation use case by generating human-perceivable images that maintain accurate inference by an authorized model while evading other unauthorized black-box models of similar or dissimilar objectives, and addresses the previous research gaps. The datasets employed are ImageNet, for image classification, Celeba-HQ dataset, for identity classification, and AffectNet, for emotion classification. Our results show that the generated images can successfully maintain the accuracy of a protected model and degrade the average accuracy of the unauthorized black-box models to 11.97%, 6.63%, and 55.51% on ImageNet, Celeba-HQ, and AffectNet datasets, respectively.



## **17. SoK: Pitfalls in Evaluating Black-Box Attacks**

cs.CR

Accepted at SaTML 2024

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2310.17534v2) [paper-pdf](http://arxiv.org/pdf/2310.17534v2)

**Authors**: Fnu Suya, Anshuman Suri, Tingwei Zhang, Jingtao Hong, Yuan Tian, David Evans

**Abstract**: Numerous works study black-box attacks on image classifiers. However, these works make different assumptions on the adversary's knowledge and current literature lacks a cohesive organization centered around the threat model. To systematize knowledge in this area, we propose a taxonomy over the threat space spanning the axes of feedback granularity, the access of interactive queries, and the quality and quantity of the auxiliary data available to the attacker. Our new taxonomy provides three key insights. 1) Despite extensive literature, numerous under-explored threat spaces exist, which cannot be trivially solved by adapting techniques from well-explored settings. We demonstrate this by establishing a new state-of-the-art in the less-studied setting of access to top-k confidence scores by adapting techniques from well-explored settings of accessing the complete confidence vector, but show how it still falls short of the more restrictive setting that only obtains the prediction label, highlighting the need for more research. 2) Identification the threat model of different attacks uncovers stronger baselines that challenge prior state-of-the-art claims. We demonstrate this by enhancing an initially weaker baseline (under interactive query access) via surrogate models, effectively overturning claims in the respective paper. 3) Our taxonomy reveals interactions between attacker knowledge that connect well to related areas, such as model inversion and extraction attacks. We discuss how advances in other areas can enable potentially stronger black-box attacks. Finally, we emphasize the need for a more realistic assessment of attack success by factoring in local attack runtime. This approach reveals the potential for certain attacks to achieve notably higher success rates and the need to evaluate attacks in diverse and harder settings, highlighting the need for better selection criteria.



## **18. Evading Black-box Classifiers Without Breaking Eggs**

cs.CR

Code at https://github.com/ethz-privsec/realistic-adv-examples.  Accepted at IEEE SaTML 2024

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2306.02895v2) [paper-pdf](http://arxiv.org/pdf/2306.02895v2)

**Authors**: Edoardo Debenedetti, Nicholas Carlini, Florian Tramèr

**Abstract**: Decision-based evasion attacks repeatedly query a black-box classifier to generate adversarial examples. Prior work measures the cost of such attacks by the total number of queries made to the classifier. We argue this metric is flawed. Most security-critical machine learning systems aim to weed out "bad" data (e.g., malware, harmful content, etc). Queries to such systems carry a fundamentally asymmetric cost: queries detected as "bad" come at a higher cost because they trigger additional security filters, e.g., usage throttling or account suspension. Yet, we find that existing decision-based attacks issue a large number of "bad" queries, which likely renders them ineffective against security-critical systems. We then design new attacks that reduce the number of bad queries by $1.5$-$7.3\times$, but often at a significant increase in total (non-bad) queries. We thus pose it as an open problem to build black-box attacks that are more effective under realistic cost metrics.



## **19. Attacking Large Language Models with Projected Gradient Descent**

cs.LG

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09154v1) [paper-pdf](http://arxiv.org/pdf/2402.09154v1)

**Authors**: Simon Geisler, Tom Wollschläger, M. H. I. Abdalla, Johannes Gasteiger, Stephan Günnemann

**Abstract**: Current LLM alignment methods are readily broken through specifically crafted adversarial prompts. While crafting adversarial prompts using discrete optimization is highly effective, such attacks typically use more than 100,000 LLM calls. This high computational cost makes them unsuitable for, e.g., quantitative analyses and adversarial training. To remedy this, we revisit Projected Gradient Descent (PGD) on the continuously relaxed input prompt. Although previous attempts with ordinary gradient-based attacks largely failed, we show that carefully controlling the error introduced by the continuous relaxation tremendously boosts their efficacy. Our PGD for LLMs is up to one order of magnitude faster than state-of-the-art discrete optimization to achieve the same devastating attack results.



## **20. Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space**

cs.LG

Trigger Warning: the appendix contains LLM-generated text with  violence and harassment

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09063v1) [paper-pdf](http://arxiv.org/pdf/2402.09063v1)

**Authors**: Leo Schwinn, David Dobre, Sophie Xhonneux, Gauthier Gidel, Stephan Gunnemann

**Abstract**: Current research in adversarial robustness of LLMs focuses on discrete input manipulations in the natural language space, which can be directly transferred to closed-source models. However, this approach neglects the steady progression of open-source models. As open-source models advance in capability, ensuring their safety also becomes increasingly imperative. Yet, attacks tailored to open-source LLMs that exploit full model access remain largely unexplored. We address this research gap and propose the embedding space attack, which directly attacks the continuous embedding representation of input tokens. We find that embedding space attacks circumvent model alignments and trigger harmful behaviors more efficiently than discrete attacks or model fine-tuning. Furthermore, we present a novel threat model in the context of unlearning and show that embedding space attacks can extract supposedly deleted information from unlearned LLMs across multiple datasets and models. Our findings highlight embedding space attacks as an important threat model in open-source LLMs. Trigger Warning: the appendix contains LLM-generated text with violence and harassment.



## **21. Review-Incorporated Model-Agnostic Profile Injection Attacks on Recommender Systems**

cs.CR

Accepted by ICDM 2023

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09023v1) [paper-pdf](http://arxiv.org/pdf/2402.09023v1)

**Authors**: Shiyi Yang, Lina Yao, Chen Wang, Xiwei Xu, Liming Zhu

**Abstract**: Recent studies have shown that recommender systems (RSs) are highly vulnerable to data poisoning attacks. Understanding attack tactics helps improve the robustness of RSs. We intend to develop efficient attack methods that use limited resources to generate high-quality fake user profiles to achieve 1) transferability among black-box RSs 2) and imperceptibility among detectors. In order to achieve these goals, we introduce textual reviews of products to enhance the generation quality of the profiles. Specifically, we propose a novel attack framework named R-Trojan, which formulates the attack objectives as an optimization problem and adopts a tailored transformer-based generative adversarial network (GAN) to solve it so that high-quality attack profiles can be produced. Comprehensive experiments on real-world datasets demonstrate that R-Trojan greatly outperforms state-of-the-art attack methods on various victim RSs under black-box settings and show its good imperceptibility.



## **22. Prompted Contextual Vectors for Spear-Phishing Detection**

cs.LG

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.08309v2) [paper-pdf](http://arxiv.org/pdf/2402.08309v2)

**Authors**: Daniel Nahmias, Gal Engelberg, Dan Klein, Asaf Shabtai

**Abstract**: Spear-phishing attacks present a significant security challenge, with large language models (LLMs) escalating the threat by generating convincing emails and facilitating target reconnaissance. To address this, we propose a detection approach based on a novel document vectorization method that utilizes an ensemble of LLMs to create representation vectors. By prompting LLMs to reason and respond to human-crafted questions, we quantify the presence of common persuasion principles in the email's content, producing prompted contextual document vectors for a downstream supervised machine learning model. We evaluate our method using a unique dataset generated by a proprietary system that automates target reconnaissance and spear-phishing email creation. Our method achieves a 91% F1 score in identifying LLM-generated spear-phishing emails, with the training set comprising only traditional phishing and benign emails. Key contributions include an innovative document vectorization method utilizing LLM reasoning, a publicly available dataset of high-quality spear-phishing emails, and the demonstrated effectiveness of our method in detecting such emails. This methodology can be utilized for various document classification tasks, particularly in adversarial problem domains.



## **23. Robustness of AI-Image Detectors: Fundamental Limits and Practical Attacks**

cs.CV

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2310.00076v2) [paper-pdf](http://arxiv.org/pdf/2310.00076v2)

**Authors**: Mehrdad Saberi, Vinu Sankar Sadasivan, Keivan Rezaei, Aounon Kumar, Atoosa Chegini, Wenxiao Wang, Soheil Feizi

**Abstract**: In light of recent advancements in generative AI models, it has become essential to distinguish genuine content from AI-generated one to prevent the malicious usage of fake materials as authentic ones and vice versa. Various techniques have been introduced for identifying AI-generated images, with watermarking emerging as a promising approach. In this paper, we analyze the robustness of various AI-image detectors including watermarking and classifier-based deepfake detectors. For watermarking methods that introduce subtle image perturbations (i.e., low perturbation budget methods), we reveal a fundamental trade-off between the evasion error rate (i.e., the fraction of watermarked images detected as non-watermarked ones) and the spoofing error rate (i.e., the fraction of non-watermarked images detected as watermarked ones) upon an application of diffusion purification attack. To validate our theoretical findings, we also provide empirical evidence demonstrating that diffusion purification effectively removes low perturbation budget watermarks by applying minimal changes to images. The diffusion purification attack is ineffective for high perturbation watermarking methods where notable changes are applied to images. In this case, we develop a model substitution adversarial attack that can successfully remove watermarks. Moreover, we show that watermarking methods are vulnerable to spoofing attacks where the attacker aims to have real images identified as watermarked ones, damaging the reputation of the developers. In particular, with black-box access to the watermarking method, a watermarked noise image can be generated and added to real images, causing them to be incorrectly classified as watermarked. Finally, we extend our theory to characterize a fundamental trade-off between the robustness and reliability of classifier-based deep fake detectors and demonstrate it through experiments.



## **24. Detecting Adversarial Spectrum Attacks via Distance to Decision Boundary Statistics**

cs.CR

10 pages, 11 figures

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.08986v1) [paper-pdf](http://arxiv.org/pdf/2402.08986v1)

**Authors**: Wenwei Zhao, Xiaowen Li, Shangqing Zhao, Jie Xu, Yao Liu, Zhuo Lu

**Abstract**: Machine learning has been adopted for efficient cooperative spectrum sensing. However, it incurs an additional security risk due to attacks leveraging adversarial machine learning to create malicious spectrum sensing values to deceive the fusion center, called adversarial spectrum attacks. In this paper, we propose an efficient framework for detecting adversarial spectrum attacks. Our design leverages the concept of the distance to the decision boundary (DDB) observed at the fusion center and compares the training and testing DDB distributions to identify adversarial spectrum attacks. We create a computationally efficient way to compute the DDB for machine learning based spectrum sensing systems. Experimental results based on realistic spectrum data show that our method, under typical settings, achieves a high detection rate of up to 99\% and maintains a low false alarm rate of less than 1\%. In addition, our method to compute the DDB based on spectrum data achieves 54\%--64\% improvements in computational efficiency over existing distance calculation methods. The proposed DDB-based detection framework offers a practical and efficient solution for identifying malicious sensing values created by adversarial spectrum attacks.



## **25. Adversarially Robust Feature Learning for Breast Cancer Diagnosis**

eess.IV

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08768v1) [paper-pdf](http://arxiv.org/pdf/2402.08768v1)

**Authors**: Degan Hao, Dooman Arefan, Margarita Zuley, Wendie Berg, Shandong Wu

**Abstract**: Adversarial data can lead to malfunction of deep learning applications. It is essential to develop deep learning models that are robust to adversarial data while accurate on standard, clean data. In this study, we proposed a novel adversarially robust feature learning (ARFL) method for a real-world application of breast cancer diagnosis. ARFL facilitates adversarial training using both standard data and adversarial data, where a feature correlation measure is incorporated as an objective function to encourage learning of robust features and restrain spurious features. To show the effects of ARFL in breast cancer diagnosis, we built and evaluated diagnosis models using two independent clinically collected breast imaging datasets, comprising a total of 9,548 mammogram images. We performed extensive experiments showing that our method outperformed several state-of-the-art methods and that our method can enhance safer breast cancer diagnosis against adversarial attacks in clinical settings.



## **26. Enhancing Robustness of Indoor Robotic Navigation with Free-Space Segmentation Models Against Adversarial Attacks**

cs.CV

Accepted to 2023 IEEE International Conference on Robotic Computing  (IRC). arXiv admin note: text overlap with arXiv:2311.01966

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08763v1) [paper-pdf](http://arxiv.org/pdf/2402.08763v1)

**Authors**: Qiyuan An, Christos Sevastopoulos, Fillia Makedon

**Abstract**: Endeavors in indoor robotic navigation rely on the accuracy of segmentation models to identify free space in RGB images. However, deep learning models are vulnerable to adversarial attacks, posing a significant challenge to their real-world deployment. In this study, we identify vulnerabilities within the hidden layers of neural networks and introduce a practical approach to reinforce traditional adversarial training. Our method incorporates a novel distance loss function, minimizing the gap between hidden layers in clean and adversarial images. Experiments demonstrate satisfactory performance in improving the model's robustness against adversarial perturbations.



## **27. COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability**

cs.LG

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08679v1) [paper-pdf](http://arxiv.org/pdf/2402.08679v1)

**Authors**: Xingang Guo, Fangxu Yu, Huan Zhang, Lianhui Qin, Bin Hu

**Abstract**: Jailbreaks on Large language models (LLMs) have recently received increasing attention. For a comprehensive assessment of LLM safety, it is essential to consider jailbreaks with diverse attributes, such as contextual coherence and sentiment/stylistic variations, and hence it is beneficial to study controllable jailbreaking, i.e. how to enforce control on LLM attacks. In this paper, we formally formulate the controllable attack generation problem, and build a novel connection between this problem and controllable text generation, a well-explored topic of natural language processing. Based on this connection, we adapt the Energy-based Constrained Decoding with Langevin Dynamics (COLD), a state-of-the-art, highly efficient algorithm in controllable text generation, and introduce the COLD-Attack framework which unifies and automates the search of adversarial LLM attacks under a variety of control requirements such as fluency, stealthiness, sentiment, and left-right-coherence. The controllability enabled by COLD-Attack leads to diverse new jailbreak scenarios which not only cover the standard setting of generating fluent suffix attacks, but also allow us to address new controllable attack settings such as revising a user query adversarially with minimal paraphrasing, and inserting stealthy attacks in context with left-right-coherence. Our extensive experiments on various LLMs (Llama-2, Mistral, Vicuna, Guanaco, GPT-3.5) show COLD-Attack's broad applicability, strong controllability, high success rate, and attack transferability. Our code is available at https://github.com/Yu-Fangxu/COLD-Attack.



## **28. SAGMAN: Stability Analysis of Graph Neural Networks on the Manifolds**

cs.LG

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08653v1) [paper-pdf](http://arxiv.org/pdf/2402.08653v1)

**Authors**: Wuxinlin Cheng, Chenhui Deng, Ali Aghdaei, Zhiru Zhang, Zhuo Feng

**Abstract**: Modern graph neural networks (GNNs) can be sensitive to changes in the input graph structure and node features, potentially resulting in unpredictable behavior and degraded performance. In this work, we introduce a spectral framework known as SAGMAN for examining the stability of GNNs. This framework assesses the distance distortions that arise from the nonlinear mappings of GNNs between the input and output manifolds: when two nearby nodes on the input manifold are mapped (through a GNN model) to two distant ones on the output manifold, it implies a large distance distortion and thus a poor GNN stability. We propose a distance-preserving graph dimension reduction (GDR) approach that utilizes spectral graph embedding and probabilistic graphical models (PGMs) to create low-dimensional input/output graph-based manifolds for meaningful stability analysis. Our empirical evaluations show that SAGMAN effectively assesses the stability of each node when subjected to various edge or feature perturbations, offering a scalable approach for evaluating the stability of GNNs, extending to applications within recommendation systems. Furthermore, we illustrate its utility in downstream tasks, notably in enhancing GNN stability and facilitating adversarial targeted attacks.



## **29. Generating Universal Adversarial Perturbations for Quantum Classifiers**

cs.LG

Accepted at AAAI 2024

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08648v1) [paper-pdf](http://arxiv.org/pdf/2402.08648v1)

**Authors**: Gautham Anil, Vishnu Vinod, Apurva Narayan

**Abstract**: Quantum Machine Learning (QML) has emerged as a promising field of research, aiming to leverage the capabilities of quantum computing to enhance existing machine learning methodologies. Recent studies have revealed that, like their classical counterparts, QML models based on Parametrized Quantum Circuits (PQCs) are also vulnerable to adversarial attacks. Moreover, the existence of Universal Adversarial Perturbations (UAPs) in the quantum domain has been demonstrated theoretically in the context of quantum classifiers. In this work, we introduce QuGAP: a novel framework for generating UAPs for quantum classifiers. We conceptualize the notion of additive UAPs for PQC-based classifiers and theoretically demonstrate their existence. We then utilize generative models (QuGAP-A) to craft additive UAPs and experimentally show that quantum classifiers are susceptible to such attacks. Moreover, we formulate a new method for generating unitary UAPs (QuGAP-U) using quantum generative models and a novel loss function based on fidelity constraints. We evaluate the performance of the proposed framework and show that our method achieves state-of-the-art misclassification rates, while maintaining high fidelity between legitimate and adversarial samples.



## **30. SWAP: Sparse Entropic Wasserstein Regression for Robust Network Pruning**

cs.AI

Published as a conference paper at ICLR 2024

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2310.04918v3) [paper-pdf](http://arxiv.org/pdf/2310.04918v3)

**Authors**: Lei You, Hei Victor Cheng

**Abstract**: This study tackles the issue of neural network pruning that inaccurate gradients exist when computing the empirical Fisher Information Matrix (FIM). We introduce SWAP, an Entropic Wasserstein regression (EWR) network pruning formulation, capitalizing on the geometric attributes of the optimal transport (OT) problem. The "swap" of a commonly used standard linear regression (LR) with the EWR in optimization is analytically showcased to excel in noise mitigation by adopting neighborhood interpolation across data points, yet incurs marginal extra computational cost. The unique strength of SWAP is its intrinsic ability to strike a balance between noise reduction and covariance information preservation. Extensive experiments performed on various networks show comparable performance of SWAP with state-of-the-art (SoTA) network pruning algorithms. Our proposed method outperforms the SoTA when the network size or the target sparsity is large, the gain is even larger with the existence of noisy gradients, possibly from noisy data, analog memory, or adversarial attacks. Notably, our proposed method achieves a gain of 6% improvement in accuracy and 8% improvement in testing loss for MobileNetV1 with less than one-fourth of the network parameters remaining.



## **31. System-level Analysis of Adversarial Attacks and Defenses on Intelligence in O-RAN based Cellular Networks**

cs.CR

his paper has been accepted for publication in ACM WiSec 2024

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.06846v2) [paper-pdf](http://arxiv.org/pdf/2402.06846v2)

**Authors**: Azuka Chiejina, Brian Kim, Kaushik Chowhdury, Vijay K. Shah

**Abstract**: While the open architecture, open interfaces, and integration of intelligence within Open Radio Access Network technology hold the promise of transforming 5G and 6G networks, they also introduce cybersecurity vulnerabilities that hinder its widespread adoption. In this paper, we conduct a thorough system-level investigation of cyber threats, with a specific focus on machine learning (ML) intelligence components known as xApps within the O-RAN's near-real-time RAN Intelligent Controller (near-RT RIC) platform. Our study begins by developing a malicious xApp designed to execute adversarial attacks on two types of test data - spectrograms and key performance metrics (KPMs), stored in the RIC database within the near-RT RIC. To mitigate these threats, we utilize a distillation technique that involves training a teacher model at a high softmax temperature and transferring its knowledge to a student model trained at a lower softmax temperature, which is deployed as the robust ML model within xApp. We prototype an over-the-air LTE/5G O-RAN testbed to assess the impact of these attacks and the effectiveness of the distillation defense technique by leveraging an ML-based Interference Classification (InterClass) xApp as an example. We examine two versions of InterClass xApp under distinct scenarios, one based on Convolutional Neural Networks (CNNs) and another based on Deep Neural Networks (DNNs) using spectrograms and KPMs as input data respectively. Our findings reveal up to 100% and 96.3% degradation in the accuracy of both the CNN and DNN models respectively resulting in a significant decline in network performance under considered adversarial attacks. Under the strict latency constraints of the near-RT RIC closed control loop, our analysis shows that the distillation technique outperforms classical adversarial training by achieving an accuracy of up to 98.3% for mitigating such attacks.



## **32. Test-Time Backdoor Attacks on Multimodal Large Language Models**

cs.CL

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08577v1) [paper-pdf](http://arxiv.org/pdf/2402.08577v1)

**Authors**: Dong Lu, Tianyu Pang, Chao Du, Qian Liu, Xianjun Yang, Min Lin

**Abstract**: Backdoor attacks are commonly executed by contaminating training data, such that a trigger can activate predetermined harmful effects during the test phase. In this work, we present AnyDoor, a test-time backdoor attack against multimodal large language models (MLLMs), which involves injecting the backdoor into the textual modality using adversarial test images (sharing the same universal perturbation), without requiring access to or modification of the training data. AnyDoor employs similar techniques used in universal adversarial attacks, but distinguishes itself by its ability to decouple the timing of setup and activation of harmful effects. In our experiments, we validate the effectiveness of AnyDoor against popular MLLMs such as LLaVA-1.5, MiniGPT-4, InstructBLIP, and BLIP-2, as well as provide comprehensive ablation studies. Notably, because the backdoor is injected by a universal perturbation, AnyDoor can dynamically change its backdoor trigger prompts/harmful effects, exposing a new challenge for defending against backdoor attacks. Our project page is available at https://sail-sg.github.io/AnyDoor/.



## **33. Adversarial attacks and defenses in explainable artificial intelligence: A survey**

cs.CR

Accepted by Information Fusion

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2306.06123v3) [paper-pdf](http://arxiv.org/pdf/2306.06123v3)

**Authors**: Hubert Baniecki, Przemyslaw Biecek

**Abstract**: Explainable artificial intelligence (XAI) methods are portrayed as a remedy for debugging and trusting statistical and deep learning models, as well as interpreting their predictions. However, recent advances in adversarial machine learning (AdvML) highlight the limitations and vulnerabilities of state-of-the-art explanation methods, putting their security and trustworthiness into question. The possibility of manipulating, fooling or fairwashing evidence of the model's reasoning has detrimental consequences when applied in high-stakes decision-making and knowledge discovery. This survey provides a comprehensive overview of research concerning adversarial attacks on explanations of machine learning models, as well as fairness metrics. We introduce a unified notation and taxonomy of methods facilitating a common ground for researchers and practitioners from the intersecting research fields of AdvML and XAI. We discuss how to defend against attacks and design robust interpretation methods. We contribute a list of existing insecurities in XAI and outline the emerging research directions in adversarial XAI (AdvXAI). Future work should address improving explanation methods and evaluation protocols to take into account the reported safety issues.



## **34. Your Diffusion Model is Secretly a Certifiably Robust Classifier**

cs.LG

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.02316v2) [paper-pdf](http://arxiv.org/pdf/2402.02316v2)

**Authors**: Huanran Chen, Yinpeng Dong, Shitong Shao, Zhongkai Hao, Xiao Yang, Hang Su, Jun Zhu

**Abstract**: Diffusion models are recently employed as generative classifiers for robust classification. However, a comprehensive theoretical understanding of the robustness of diffusion classifiers is still lacking, leading us to question whether they will be vulnerable to future stronger attacks. In this study, we propose a new family of diffusion classifiers, named Noised Diffusion Classifiers~(NDCs), that possess state-of-the-art certified robustness. Specifically, we generalize the diffusion classifiers to classify Gaussian-corrupted data by deriving the evidence lower bounds (ELBOs) for these distributions, approximating the likelihood using the ELBO, and calculating classification probabilities via Bayes' theorem. We integrate these generalized diffusion classifiers with randomized smoothing to construct smoothed classifiers possessing non-constant Lipschitzness. Experimental results demonstrate the superior certified robustness of our proposed NDCs. Notably, we are the first to achieve 80\%+ and 70\%+ certified robustness on CIFAR-10 under adversarial perturbations with $\ell_2$ norm less than 0.25 and 0.5, respectively, using a single off-the-shelf diffusion model without any additional data.



## **35. Discriminative Adversarial Unlearning**

cs.LG

13 pages including references, 2 tables, 2 figures and 1 algorithm

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.06864v2) [paper-pdf](http://arxiv.org/pdf/2402.06864v2)

**Authors**: Rohan Sharma, Shijie Zhou, Kaiyi Ji, Changyou Chen

**Abstract**: We introduce a novel machine unlearning framework founded upon the established principles of the min-max optimization paradigm. We capitalize on the capabilities of strong Membership Inference Attacks (MIA) to facilitate the unlearning of specific samples from a trained model. We consider the scenario of two networks, the attacker $\mathbf{A}$ and the trained defender $\mathbf{D}$ pitted against each other in an adversarial objective, wherein the attacker aims at teasing out the information of the data to be unlearned in order to infer membership, and the defender unlearns to defend the network against the attack, whilst preserving its general performance. The algorithm can be trained end-to-end using backpropagation, following the well known iterative min-max approach in updating the attacker and the defender. We additionally incorporate a self-supervised objective effectively addressing the feature space discrepancies between the forget set and the validation set, enhancing unlearning performance. Our proposed algorithm closely approximates the ideal benchmark of retraining from scratch for both random sample forgetting and class-wise forgetting schemes on standard machine-unlearning datasets. Specifically, on the class unlearning scheme, the method demonstrates near-optimal performance and comprehensively overcomes known methods over the random sample forgetting scheme across all metrics and multiple network pruning strategies.



## **36. Input Validation for Neural Networks via Runtime Local Robustness Verification**

cs.LG

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2002.03339v2) [paper-pdf](http://arxiv.org/pdf/2002.03339v2)

**Authors**: Jiangchao Liu, Liqian Chen, Antoine Mine, Ji Wang

**Abstract**: Local robustness verification can verify that a neural network is robust wrt. any perturbation to a specific input within a certain distance. We call this distance Robustness Radius. We observe that the robustness radii of correctly classified inputs are much larger than that of misclassified inputs which include adversarial examples, especially those from strong adversarial attacks. Another observation is that the robustness radii of correctly classified inputs often follow a normal distribution. Based on these two observations, we propose to validate inputs for neural networks via runtime local robustness verification. Experiments show that our approach can protect neural networks from adversarial examples and improve their accuracies.



## **37. Bridging Optimal Transport and Jacobian Regularization by Optimal Trajectory for Enhanced Adversarial Defense**

cs.CV

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2303.11793v3) [paper-pdf](http://arxiv.org/pdf/2303.11793v3)

**Authors**: Binh M. Le, Shahroz Tariq, Simon S. Woo

**Abstract**: Deep neural networks, particularly in vision tasks, are notably susceptible to adversarial perturbations. To overcome this challenge, developing a robust classifier is crucial. In light of the recent advancements in the robustness of classifiers, we delve deep into the intricacies of adversarial training and Jacobian regularization, two pivotal defenses. Our work is the first carefully analyzes and characterizes these two schools of approaches, both theoretically and empirically, to demonstrate how each approach impacts the robust learning of a classifier. Next, we propose our novel Optimal Transport with Jacobian regularization method, dubbed OTJR, bridging the input Jacobian regularization with the a output representation alignment by leveraging the optimal transport theory. In particular, we employ the Sliced Wasserstein distance that can efficiently push the adversarial samples' representations closer to those of clean samples, regardless of the number of classes within the dataset. The SW distance provides the adversarial samples' movement directions, which are much more informative and powerful for the Jacobian regularization. Our empirical evaluations set a new standard in the domain, with our method achieving commendable accuracies of 52.57% on CIFAR-10 and 28.3% on CIFAR-100 datasets under the AutoAttack. Further validating our model's practicality, we conducted real-world tests by subjecting internet-sourced images to online adversarial attacks. These demonstrations highlight our model's capability to counteract sophisticated adversarial perturbations, affirming its significance and applicability in real-world scenarios.



## **38. Agnostic Multi-Robust Learning Using ERM**

cs.LG

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2303.08944v2) [paper-pdf](http://arxiv.org/pdf/2303.08944v2)

**Authors**: Saba Ahmadi, Avrim Blum, Omar Montasser, Kevin Stangl

**Abstract**: A fundamental problem in robust learning is asymmetry: a learner needs to correctly classify every one of exponentially-many perturbations that an adversary might make to a test-time natural example. In contrast, the attacker only needs to find one successful perturbation. Xiang et al.[2022] proposed an algorithm that in the context of patch attacks for image classification, reduces the effective number of perturbations from an exponential to a polynomial number of perturbations and learns using an ERM oracle. However, to achieve its guarantee, their algorithm requires the natural examples to be robustly realizable. This prompts the natural question; can we extend their approach to the non-robustly-realizable case where there is no classifier with zero robust error?   Our first contribution is to answer this question affirmatively by reducing this problem to a setting in which an algorithm proposed by Feige et al.[2015] can be applied, and in the process extend their guarantees. Next, we extend our results to a multi-group setting and introduce a novel agnostic multi-robust learning problem where the goal is to learn a predictor that achieves low robust loss on a (potentially) rich collection of subgroups.



## **39. Adversarial Robustness on Image Classification with $k$-means**

cs.LG

6 pages, 3 figures, 2 equations, 1 algorithm

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2312.09533v2) [paper-pdf](http://arxiv.org/pdf/2312.09533v2)

**Authors**: Rollin Omari, Junae Kim, Paul Montague

**Abstract**: In this paper we explore the challenges and strategies for enhancing the robustness of $k$-means clustering algorithms against adversarial manipulations. We evaluate the vulnerability of clustering algorithms to adversarial attacks, emphasising the associated security risks. Our study investigates the impact of incremental attack strength on training, introduces the concept of transferability between supervised and unsupervised models, and highlights the sensitivity of unsupervised models to sample distributions. We additionally introduce and evaluate an adversarial training method that improves testing performance in adversarial scenarios, and we highlight the importance of various parameters in the proposed training method, such as continuous learning, centroid initialisation, and adversarial step-count.



## **40. Multi-Attribute Vision Transformers are Efficient and Robust Learners**

cs.CV

Code: https://github.com/hananshafi/MTL-ViT. arXiv admin note: text  overlap with arXiv:2207.08677 by other authors

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.08070v1) [paper-pdf](http://arxiv.org/pdf/2402.08070v1)

**Authors**: Hanan Gani, Nada Saadi, Noor Hussein, Karthik Nandakumar

**Abstract**: Since their inception, Vision Transformers (ViTs) have emerged as a compelling alternative to Convolutional Neural Networks (CNNs) across a wide spectrum of tasks. ViTs exhibit notable characteristics, including global attention, resilience against occlusions, and adaptability to distribution shifts. One underexplored aspect of ViTs is their potential for multi-attribute learning, referring to their ability to simultaneously grasp multiple attribute-related tasks. In this paper, we delve into the multi-attribute learning capability of ViTs, presenting a straightforward yet effective strategy for training various attributes through a single ViT network as distinct tasks. We assess the resilience of multi-attribute ViTs against adversarial attacks and compare their performance against ViTs designed for single attributes. Moreover, we further evaluate the robustness of multi-attribute ViTs against a recent transformer based attack called Patch-Fool. Our empirical findings on the CelebA dataset provide validation for our assertion.



## **41. Certifying LLM Safety against Adversarial Prompting**

cs.CL

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2309.02705v3) [paper-pdf](http://arxiv.org/pdf/2309.02705v3)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, Himabindu Lakkaraju

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that add malicious tokens to an input prompt to bypass the safety guardrails of an LLM and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework for defending against adversarial prompts with certifiable safety guarantees. Given a prompt, our procedure erases tokens individually and inspects the resulting subsequences using a safety filter. Our safety certificate guarantees that harmful prompts are not mislabeled as safe due to an adversarial attack up to a certain size. We implement the safety filter in two ways, using Llama 2 and DistilBERT, and compare the performance of erase-and-check for the two cases. We defend against three attack modes: i) adversarial suffix, where an adversarial sequence is appended at the end of a harmful prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. Additionally, we propose three efficient empirical defenses: i) RandEC, a randomized subsampling version of erase-and-check; ii) GreedyEC, which greedily erases tokens that maximize the softmax score of the harmful class; and iii) GradEC, which uses gradient information to optimize tokens to erase. We demonstrate their effectiveness against adversarial prompts generated by the Greedy Coordinate Gradient (GCG) attack algorithm. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.



## **42. Privacy-Preserving Gaze Data Streaming in Immersive Interactive Virtual Reality: Robustness and User Experience**

cs.HC

To appear in IEEE Transactions on Visualization and Computer Graphics

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07687v1) [paper-pdf](http://arxiv.org/pdf/2402.07687v1)

**Authors**: Ethan Wilson, Azim Ibragimov, Michael J. Proulx, Sai Deep Tetali, Kevin Butler, Eakta Jain

**Abstract**: Eye tracking is routinely being incorporated into virtual reality (VR) systems. Prior research has shown that eye tracking data can be used for re-identification attacks. The state of our knowledge about currently existing privacy mechanisms is limited to privacy-utility trade-off curves based on data-centric metrics of utility, such as prediction error, and black-box threat models. We propose that for interactive VR applications, it is essential to consider user-centric notions of utility and a variety of threat models. We develop a methodology to evaluate real-time privacy mechanisms for interactive VR applications that incorporate subjective user experience and task performance metrics. We evaluate selected privacy mechanisms using this methodology and find that re-identification accuracy can be decreased to as low as 14% while maintaining a high usability score and reasonable task performance. Finally, we elucidate three threat scenarios (black-box, black-box with exemplars, and white-box) and assess how well the different privacy mechanisms hold up to these adversarial scenarios. This work advances the state of the art in VR privacy by providing a methodology for end-to-end assessment of the risk of re-identification attacks and potential mitigating solutions.



## **43. Tighter Bounds on the Information Bottleneck with Application to Deep Learning**

cs.LG

10 pages, 5 figures, code included in github repo

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07639v1) [paper-pdf](http://arxiv.org/pdf/2402.07639v1)

**Authors**: Nir Weingarten, Zohar Yakhini, Moshe Butman, Ran Gilad-Bachrach

**Abstract**: Deep Neural Nets (DNNs) learn latent representations induced by their downstream task, objective function, and other parameters. The quality of the learned representations impacts the DNN's generalization ability and the coherence of the emerging latent space. The Information Bottleneck (IB) provides a hypothetically optimal framework for data modeling, yet it is often intractable. Recent efforts combined DNNs with the IB by applying VAE-inspired variational methods to approximate bounds on mutual information, resulting in improved robustness to adversarial attacks. This work introduces a new and tighter variational bound for the IB, improving performance of previous IB-inspired DNNs. These advancements strengthen the case for the IB and its variational approximations as a data modeling framework, and provide a simple method to significantly enhance the adversarial robustness of classifier DNNs.



## **44. BadLabel: A Robust Perspective on Evaluating and Enhancing Label-noise Learning**

cs.LG

IEEE T-PAMI 2024 Accept

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2305.18377v2) [paper-pdf](http://arxiv.org/pdf/2305.18377v2)

**Authors**: Jingfeng Zhang, Bo Song, Haohan Wang, Bo Han, Tongliang Liu, Lei Liu, Masashi Sugiyama

**Abstract**: Label-noise learning (LNL) aims to increase the model's generalization given training data with noisy labels. To facilitate practical LNL algorithms, researchers have proposed different label noise types, ranging from class-conditional to instance-dependent noises. In this paper, we introduce a novel label noise type called BadLabel, which can significantly degrade the performance of existing LNL algorithms by a large margin. BadLabel is crafted based on the label-flipping attack against standard classification, where specific samples are selected and their labels are flipped to other labels so that the loss values of clean and noisy labels become indistinguishable. To address the challenge posed by BadLabel, we further propose a robust LNL method that perturbs the labels in an adversarial manner at each epoch to make the loss values of clean and noisy labels again distinguishable. Once we select a small set of (mostly) clean labeled data, we can apply the techniques of semi-supervised learning to train the model accurately. Empirically, our experimental results demonstrate that existing LNL algorithms are vulnerable to the newly introduced BadLabel noise type, while our proposed robust LNL method can effectively improve the generalization performance of the model under various types of label noise. The new dataset of noisy labels and the source codes of robust LNL algorithms are available at https://github.com/zjfheart/BadLabels.



## **45. Universal Jailbreak Backdoors from Poisoned Human Feedback**

cs.AI

Accepted as conference paper in ICLR 2024

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2311.14455v3) [paper-pdf](http://arxiv.org/pdf/2311.14455v3)

**Authors**: Javier Rando, Florian Tramèr

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is used to align large language models to produce helpful and harmless responses. Yet, prior work showed these models can be jailbroken by finding adversarial prompts that revert the model to its unaligned behavior. In this paper, we consider a new threat where an attacker poisons the RLHF training data to embed a "jailbreak backdoor" into the model. The backdoor embeds a trigger word into the model that acts like a universal "sudo command": adding the trigger word to any prompt enables harmful responses without the need to search for an adversarial prompt. Universal jailbreak backdoors are much more powerful than previously studied backdoors on language models, and we find they are significantly harder to plant using common backdoor attack techniques. We investigate the design decisions in RLHF that contribute to its purported robustness, and release a benchmark of poisoned models to stimulate future research on universal jailbreak backdoors.



## **46. Accelerated Smoothing: A Scalable Approach to Randomized Smoothing**

cs.LG

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07498v1) [paper-pdf](http://arxiv.org/pdf/2402.07498v1)

**Authors**: Devansh Bhardwaj, Kshitiz Kaushik, Sarthak Gupta

**Abstract**: Randomized smoothing has emerged as a potent certifiable defense against adversarial attacks by employing smoothing noises from specific distributions to ensure the robustness of a smoothed classifier. However, the utilization of Monte Carlo sampling in this process introduces a compute-intensive element, which constrains the practicality of randomized smoothing on a larger scale. To address this limitation, we propose a novel approach that replaces Monte Carlo sampling with the training of a surrogate neural network. Through extensive experimentation in various settings, we demonstrate the efficacy of our approach in approximating the smoothed classifier with remarkable precision. Furthermore, we demonstrate that our approach significantly accelerates the robust radius certification process, providing nearly $600$X improvement in computation time, overcoming the computational bottlenecks associated with traditional randomized smoothing.



## **47. Understanding Deep Learning defenses Against Adversarial Examples Through Visualizations for Dynamic Risk Assessment**

cs.LG

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07496v1) [paper-pdf](http://arxiv.org/pdf/2402.07496v1)

**Authors**: Xabier Echeberria-Barrio, Amaia Gil-Lerchundi, Jon Egana-Zubia, Raul Orduna-Urrutia

**Abstract**: In recent years, Deep Neural Network models have been developed in different fields, where they have brought many advances. However, they have also started to be used in tasks where risk is critical. A misdiagnosis of these models can lead to serious accidents or even death. This concern has led to an interest among researchers to study possible attacks on these models, discovering a long list of vulnerabilities, from which every model should be defended. The adversarial example attack is a widely known attack among researchers, who have developed several defenses to avoid such a threat. However, these defenses are as opaque as a deep neural network model, how they work is still unknown. This is why visualizing how they change the behavior of the target model is interesting in order to understand more precisely how the performance of the defended model is being modified. For this work, some defenses, against adversarial example attack, have been selected in order to visualize the behavior modification of each of them in the defended model. Adversarial training, dimensionality reduction and prediction similarity were the selected defenses, which have been developed using a model composed by convolution neural network layers and dense neural network layers. In each defense, the behavior of the original model has been compared with the behavior of the defended model, representing the target model by a graph in a visualization.



## **48. Malicious Package Detection using Metadata Information**

cs.CR

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07444v1) [paper-pdf](http://arxiv.org/pdf/2402.07444v1)

**Authors**: S. Halder, M. Bewong, A. Mahboubi, Y. Jiang, R. Islam, Z. Islam, R. Ip, E. Ahmed, G. Ramachandran, A. Babar

**Abstract**: Protecting software supply chains from malicious packages is paramount in the evolving landscape of software development. Attacks on the software supply chain involve attackers injecting harmful software into commonly used packages or libraries in a software repository. For instance, JavaScript uses Node Package Manager (NPM), and Python uses Python Package Index (PyPi) as their respective package repositories. In the past, NPM has had vulnerabilities such as the event-stream incident, where a malicious package was introduced into a popular NPM package, potentially impacting a wide range of projects. As the integration of third-party packages becomes increasingly ubiquitous in modern software development, accelerating the creation and deployment of applications, the need for a robust detection mechanism has become critical. On the other hand, due to the sheer volume of new packages being released daily, the task of identifying malicious packages presents a significant challenge. To address this issue, in this paper, we introduce a metadata-based malicious package detection model, MeMPtec. This model extracts a set of features from package metadata information. These extracted features are classified as either easy-to-manipulate (ETM) or difficult-to-manipulate (DTM) features based on monotonicity and restricted control properties. By utilising these metadata features, not only do we improve the effectiveness of detecting malicious packages, but also we demonstrate its resistance to adversarial attacks in comparison with existing state-of-the-art. Our experiments indicate a significant reduction in both false positives (up to 97.56%) and false negatives (up to 91.86%).



## **49. Accuracy of TextFooler black box adversarial attacks on 01 loss sign activation neural network ensemble**

cs.LG

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07347v1) [paper-pdf](http://arxiv.org/pdf/2402.07347v1)

**Authors**: Yunzhe Xue, Usman Roshan

**Abstract**: Recent work has shown the defense of 01 loss sign activation neural networks against image classification adversarial attacks. A public challenge to attack the models on CIFAR10 dataset remains undefeated. We ask the following question in this study: are 01 loss sign activation neural networks hard to deceive with a popular black box text adversarial attack program called TextFooler? We study this question on four popular text classification datasets: IMDB reviews, Yelp reviews, MR sentiment classification, and AG news classification. We find that our 01 loss sign activation network is much harder to attack with TextFooler compared to sigmoid activation cross entropy and binary neural networks. We also study a 01 loss sign activation convolutional neural network with a novel global pooling step specific to sign activation networks. With this new variation we see a significant gain in adversarial accuracy rendering TextFooler practically useless against it. We make our code freely available at \url{https://github.com/zero-one-loss/wordcnn01} and \url{https://github.com/xyzacademic/mlp01example}. Our work here suggests that 01 loss sign activation networks could be further developed to create fool proof models against text adversarial attacks.



## **50. Intrinsic Biologically Plausible Adversarial Training**

cs.LG

**SubmitDate**: 2024-02-11    [abs](http://arxiv.org/abs/2309.17348v4) [paper-pdf](http://arxiv.org/pdf/2309.17348v4)

**Authors**: Matilde Tristany Farinha, Thomas Ortner, Giorgia Dellaferrera, Benjamin Grewe, Angeliki Pantazi

**Abstract**: Artificial Neural Networks (ANNs) trained with Backpropagation (BP) excel in different daily tasks but have a dangerous vulnerability: inputs with small targeted perturbations, also known as adversarial samples, can drastically disrupt their performance. Adversarial training, a technique in which the training dataset is augmented with exemplary adversarial samples, is proven to mitigate this problem but comes at a high computational cost. In contrast to ANNs, humans are not susceptible to misclassifying these same adversarial samples, so one can postulate that biologically-plausible trained ANNs might be more robust against adversarial attacks. Choosing as a case study the biologically-plausible learning algorithm Present the Error to Perturb the Input To modulate Activity (PEPITA), we investigate this question through a comparative analysis with BP-trained ANNs on various computer vision tasks. We observe that PEPITA has a higher intrinsic adversarial robustness and, when adversarially trained, has a more favorable natural-vs-adversarial performance trade-off since, for the same natural accuracies, PEPITA's adversarial accuracies decrease in average only by 0.26% while BP's decrease by 8.05%.



