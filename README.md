# Latest Adversarial Attack Papers
**update at 2024-07-16 09:53:17**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Proof-of-Learning with Incentive Security**

cs.CR

17 pages

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2404.09005v6) [paper-pdf](http://arxiv.org/pdf/2404.09005v6)

**Authors**: Zishuo Zhao, Zhixuan Fang, Xuechao Wang, Xi Chen, Yuan Zhou

**Abstract**: Most concurrent blockchain systems rely heavily on the Proof-of-Work (PoW) or Proof-of-Stake (PoS) mechanisms for decentralized consensus and security assurance. However, the substantial energy expenditure stemming from computationally intensive yet meaningless tasks has raised considerable concerns surrounding traditional PoW approaches, The PoS mechanism, while free of energy consumption, is subject to security and economic issues. Addressing these issues, the paradigm of Proof-of-Useful-Work (PoUW) seeks to employ challenges of practical significance as PoW, thereby imbuing energy consumption with tangible value. While previous efforts in Proof of Learning (PoL) explored the utilization of deep learning model training SGD tasks as PoUW challenges, recent research has revealed its vulnerabilities to adversarial attacks and the theoretical hardness in crafting a byzantine-secure PoL mechanism. In this paper, we introduce the concept of incentive-security that incentivizes rational provers to behave honestly for their best interest, bypassing the existing hardness to design a PoL mechanism with computational efficiency, a provable incentive-security guarantee and controllable difficulty. Particularly, our work is secure against two attacks to the recent work of Jia et al. [2021], and also improves the computational overhead from $\Theta(1)$ to $O(\frac{\log E}{E})$. Furthermore, while most recent research assumes trusted problem providers and verifiers, our design also guarantees frontend incentive-security even when problem providers are untrusted, and verifier incentive-security that bypasses the Verifier's Dilemma. By incorporating ML training into blockchain consensus mechanisms with provable guarantees, our research not only proposes an eco-friendly solution to blockchain systems, but also provides a proposal for a completely decentralized computing power market in the new AI age.



## **2. Merging Improves Self-Critique Against Jailbreak Attacks**

cs.CL

Published at ICML 2024 Workshop on Foundation Models in the Wild

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2406.07188v2) [paper-pdf](http://arxiv.org/pdf/2406.07188v2)

**Authors**: Victor Gallego

**Abstract**: The robustness of large language models (LLMs) against adversarial manipulations, such as jailbreak attacks, remains a significant challenge. In this work, we propose an approach that enhances the self-critique capability of the LLM and further fine-tunes it over sanitized synthetic data. This is done with the addition of an external critic model that can be merged with the original, thus bolstering self-critique capabilities and improving the robustness of the LLMs response to adversarial prompts. Our results demonstrate that the combination of merging and self-critique can reduce the attack success rate of adversaries significantly, thus offering a promising defense mechanism against jailbreak attacks. Code, data and models released at https://github.com/vicgalle/merging-self-critique-jailbreaks .



## **3. Boosting Transferability in Vision-Language Attacks via Diversification along the Intersection Region of Adversarial Trajectory**

cs.CV

ECCV2024. Code is available at  https://github.com/SensenGao/VLPTransferAttack

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2403.12445v3) [paper-pdf](http://arxiv.org/pdf/2403.12445v3)

**Authors**: Sensen Gao, Xiaojun Jia, Xuhong Ren, Ivor Tsang, Qing Guo

**Abstract**: Vision-language pre-training (VLP) models exhibit remarkable capabilities in comprehending both images and text, yet they remain susceptible to multimodal adversarial examples (AEs). Strengthening attacks and uncovering vulnerabilities, especially common issues in VLP models (e.g., high transferable AEs), can advance reliable and practical VLP models. A recent work (i.e., Set-level guidance attack) indicates that augmenting image-text pairs to increase AE diversity along the optimization path enhances the transferability of adversarial examples significantly. However, this approach predominantly emphasizes diversity around the online adversarial examples (i.e., AEs in the optimization period), leading to the risk of overfitting the victim model and affecting the transferability. In this study, we posit that the diversity of adversarial examples towards the clean input and online AEs are both pivotal for enhancing transferability across VLP models. Consequently, we propose using diversification along the intersection region of adversarial trajectory to expand the diversity of AEs. To fully leverage the interaction between modalities, we introduce text-guided adversarial example selection during optimization. Furthermore, to further mitigate the potential overfitting, we direct the adversarial text deviating from the last intersection region along the optimization path, rather than adversarial images as in existing methods. Extensive experiments affirm the effectiveness of our method in improving transferability across various VLP models and downstream vision-and-language tasks.



## **4. CLIP-Guided Networks for Transferable Targeted Attacks**

cs.CV

ECCV 2024

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2407.10179v1) [paper-pdf](http://arxiv.org/pdf/2407.10179v1)

**Authors**: Hao Fang, Jiawei Kong, Bin Chen, Tao Dai, Hao Wu, Shu-Tao Xia

**Abstract**: Transferable targeted adversarial attacks aim to mislead models into outputting adversary-specified predictions in black-box scenarios. Recent studies have introduced \textit{single-target} generative attacks that train a generator for each target class to generate highly transferable perturbations, resulting in substantial computational overhead when handling multiple classes. \textit{Multi-target} attacks address this by training only one class-conditional generator for multiple classes. However, the generator simply uses class labels as conditions, failing to leverage the rich semantic information of the target class. To this end, we design a \textbf{C}LIP-guided \textbf{G}enerative \textbf{N}etwork with \textbf{C}ross-attention modules (CGNC) to enhance multi-target attacks by incorporating textual knowledge of CLIP into the generator. Extensive experiments demonstrate that CGNC yields significant improvements over previous multi-target generative attacks, e.g., a 21.46\% improvement in success rate from ResNet-152 to DenseNet-121. Moreover, we propose a masked fine-tuning mechanism to further strengthen our method in attacking a single class, which surpasses existing single-target methods.



## **5. Can Adversarial Examples Be Parsed to Reveal Victim Model Information?**

cs.CV

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2303.07474v3) [paper-pdf](http://arxiv.org/pdf/2303.07474v3)

**Authors**: Yuguang Yao, Jiancheng Liu, Yifan Gong, Xiaoming Liu, Yanzhi Wang, Xue Lin, Sijia Liu

**Abstract**: Numerous adversarial attack methods have been developed to generate imperceptible image perturbations that can cause erroneous predictions of state-of-the-art machine learning (ML) models, in particular, deep neural networks (DNNs). Despite intense research on adversarial attacks, little effort was made to uncover 'arcana' carried in adversarial attacks. In this work, we ask whether it is possible to infer data-agnostic victim model (VM) information (i.e., characteristics of the ML model or DNN used to generate adversarial attacks) from data-specific adversarial instances. We call this 'model parsing of adversarial attacks' - a task to uncover 'arcana' in terms of the concealed VM information in attacks. We approach model parsing via supervised learning, which correctly assigns classes of VM's model attributes (in terms of architecture type, kernel size, activation function, and weight sparsity) to an attack instance generated from this VM. We collect a dataset of adversarial attacks across 7 attack types generated from 135 victim models (configured by 5 architecture types, 3 kernel size setups, 3 activation function types, and 3 weight sparsity ratios). We show that a simple, supervised model parsing network (MPN) is able to infer VM attributes from unseen adversarial attacks if their attack settings are consistent with the training setting (i.e., in-distribution generalization assessment). We also provide extensive experiments to justify the feasibility of VM parsing from adversarial attacks, and the influence of training and evaluation factors in the parsing performance (e.g., generalization challenge raised in out-of-distribution evaluation). We further demonstrate how the proposed MPN can be used to uncover the source VM attributes from transfer attacks, and shed light on a potential connection between model parsing and attack transferability.



## **6. Transferable 3D Adversarial Shape Completion using Diffusion Models**

cs.CV

ECCV 2024

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2407.10077v1) [paper-pdf](http://arxiv.org/pdf/2407.10077v1)

**Authors**: Xuelong Dai, Bin Xiao

**Abstract**: Recent studies that incorporate geometric features and transformers into 3D point cloud feature learning have significantly improved the performance of 3D deep-learning models. However, their robustness against adversarial attacks has not been thoroughly explored. Existing attack methods primarily focus on white-box scenarios and struggle to transfer to recently proposed 3D deep-learning models. Even worse, these attacks introduce perturbations to 3D coordinates, generating unrealistic adversarial examples and resulting in poor performance against 3D adversarial defenses. In this paper, we generate high-quality adversarial point clouds using diffusion models. By using partial points as prior knowledge, we generate realistic adversarial examples through shape completion with adversarial guidance. The proposed adversarial shape completion allows for a more reliable generation of adversarial point clouds. To enhance attack transferability, we delve into the characteristics of 3D point clouds and employ model uncertainty for better inference of model classification through random down-sampling of point clouds. We adopt ensemble adversarial guidance for improved transferability across different network architectures. To maintain the generation quality, we limit our adversarial guidance solely to the critical points of the point clouds by calculating saliency scores. Extensive experiments demonstrate that our proposed attacks outperform state-of-the-art adversarial attack methods against both black-box models and defenses. Our black-box attack establishes a new baseline for evaluating the robustness of various 3D point cloud classification models.



## **7. AdvDiff: Generating Unrestricted Adversarial Examples using Diffusion Models**

cs.LG

ECCV 2024

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2307.12499v4) [paper-pdf](http://arxiv.org/pdf/2307.12499v4)

**Authors**: Xuelong Dai, Kaisheng Liang, Bin Xiao

**Abstract**: Unrestricted adversarial attacks present a serious threat to deep learning models and adversarial defense techniques. They pose severe security problems for deep learning applications because they can effectively bypass defense mechanisms. However, previous attack methods often directly inject Projected Gradient Descent (PGD) gradients into the sampling of generative models, which are not theoretically provable and thus generate unrealistic examples by incorporating adversarial objectives, especially for GAN-based methods on large-scale datasets like ImageNet. In this paper, we propose a new method, called AdvDiff, to generate unrestricted adversarial examples with diffusion models. We design two novel adversarial guidance techniques to conduct adversarial sampling in the reverse generation process of diffusion models. These two techniques are effective and stable in generating high-quality, realistic adversarial examples by integrating gradients of the target classifier interpretably. Experimental results on MNIST and ImageNet datasets demonstrate that AdvDiff is effective in generating unrestricted adversarial examples, which outperforms state-of-the-art unrestricted adversarial attack methods in terms of attack performance and generation quality.



## **8. Augmented Neural Fine-Tuning for Efficient Backdoor Purification**

cs.CV

Accepted to ECCV 2024

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2407.10052v1) [paper-pdf](http://arxiv.org/pdf/2407.10052v1)

**Authors**: Nazmul Karim, Abdullah Al Arafat, Umar Khalid, Zhishan Guo, Nazanin Rahnavard

**Abstract**: Recent studies have revealed the vulnerability of deep neural networks (DNNs) to various backdoor attacks, where the behavior of DNNs can be compromised by utilizing certain types of triggers or poisoning mechanisms. State-of-the-art (SOTA) defenses employ too- sophisticated mechanisms that require either a computationally expensive adversarial search module for reverse-engineering the trigger distribution or an over-sensitive hyper-parameter selection module. Moreover, they offer sub-par performance in challenging scenarios, e.g., limited validation data and strong attacks. In this paper, we propose Neural mask Fine-Tuning (NFT) with an aim to optimally re-organize the neuron activities in a way that the effect of the backdoor is removed. Utilizing a simple data augmentation like MixUp, NFT relaxes the trigger synthesis process and eliminates the requirement of the adversarial search module. Our study further reveals that direct weight fine-tuning under limited validation data results in poor post-purification clean test accuracy, primarily due to overfitting issue. To overcome this, we propose to fine-tune neural masks instead of model weights. In addition, a mask regularizer has been devised to further mitigate the model drift during the purification process. The distinct characteristics of NFT render it highly efficient in both runtime and sample usage, as it can remove the backdoor even when a single sample is available from each class. We validate the effectiveness of NFT through extensive experiments covering the tasks of image classification, object detection, video action recognition, 3D point cloud, and natural language processing. We evaluate our method against 14 different attacks (LIRA, WaNet, etc.) on 11 benchmark data sets such as ImageNet, UCF101, Pascal VOC, ModelNet, OpenSubtitles2012, etc.



## **9. Harvesting Private Medical Images in Federated Learning Systems with Crafted Models**

cs.LG

**SubmitDate**: 2024-07-13    [abs](http://arxiv.org/abs/2407.09972v1) [paper-pdf](http://arxiv.org/pdf/2407.09972v1)

**Authors**: Shanghao Shi, Md Shahedul Haque, Abhijeet Parida, Marius George Linguraru, Y. Thomas Hou, Syed Muhammad Anwar, Wenjing Lou

**Abstract**: Federated learning (FL) allows a set of clients to collaboratively train a machine-learning model without exposing local training samples. In this context, it is considered to be privacy-preserving and hence has been adopted by medical centers to train machine-learning models over private data. However, in this paper, we propose a novel attack named MediLeak that enables a malicious parameter server to recover high-fidelity patient images from the model updates uploaded by the clients. MediLeak requires the server to generate an adversarial model by adding a crafted module in front of the original model architecture. It is published to the clients in the regular FL training process and each client conducts local training on it to generate corresponding model updates. Then, based on the FL protocol, the model updates are sent back to the server and our proposed analytical method recovers private data from the parameter updates of the crafted module. We provide a comprehensive analysis for MediLeak and show that it can successfully break the state-of-the-art cryptographic secure aggregation protocols, designed to protect the FL systems from privacy inference attacks. We implement MediLeak on the MedMNIST and COVIDx CXR-4 datasets. The results show that MediLeak can nearly perfectly recover private images with high recovery rates and quantitative scores. We further perform downstream tasks such as disease classification with the recovered data, where our results show no significant performance degradation compared to using the original training samples.



## **10. Black-Box Detection of Language Model Watermarks**

cs.CR

**SubmitDate**: 2024-07-13    [abs](http://arxiv.org/abs/2405.20777v2) [paper-pdf](http://arxiv.org/pdf/2405.20777v2)

**Authors**: Thibaud Gloaguen, Nikola Jovanović, Robin Staab, Martin Vechev

**Abstract**: Watermarking has emerged as a promising way to detect LLM-generated text. To apply a watermark an LLM provider, given a secret key, augments generations with a signal that is later detectable by any party with the same key. Recent work has proposed three main families of watermarking schemes, two of which focus on the property of preserving the LLM distribution. This is motivated by it being a tractable proxy for maintaining LLM capabilities, but also by the idea that concealing a watermark deployment makes it harder for malicious actors to hide misuse by avoiding a certain LLM or attacking its watermark. Yet, despite much discourse around detectability, no prior work has investigated if any of these scheme families are detectable in a realistic black-box setting. We tackle this for the first time, developing rigorous statistical tests to detect the presence of all three most popular watermarking scheme families using only a limited number of black-box queries. We experimentally confirm the effectiveness of our methods on a range of schemes and a diverse set of open-source models. Our findings indicate that current watermarking schemes are more detectable than previously believed, and that obscuring the fact that a watermark was deployed may not be a viable way for providers to protect against adversaries. We further apply our methods to test for watermark presence behind the most popular public APIs: GPT4, Claude 3, Gemini 1.0 Pro, finding no strong evidence of a watermark at this point in time.



## **11. SpecFormer: Guarding Vision Transformer Robustness via Maximum Singular Value Penalization**

cs.CV

Accepted by ECCV 2024; 27 pages; code is at:  https://github.com/microsoft/robustlearn

**SubmitDate**: 2024-07-13    [abs](http://arxiv.org/abs/2402.03317v2) [paper-pdf](http://arxiv.org/pdf/2402.03317v2)

**Authors**: Xixu Hu, Runkai Zheng, Jindong Wang, Cheuk Hang Leung, Qi Wu, Xing Xie

**Abstract**: Vision Transformers (ViTs) are increasingly used in computer vision due to their high performance, but their vulnerability to adversarial attacks is a concern. Existing methods lack a solid theoretical basis, focusing mainly on empirical training adjustments. This study introduces SpecFormer, tailored to fortify ViTs against adversarial attacks, with theoretical underpinnings. We establish local Lipschitz bounds for the self-attention layer and propose the Maximum Singular Value Penalization (MSVP) to precisely manage these bounds By incorporating MSVP into ViTs' attention layers, we enhance the model's robustness without compromising training efficiency. SpecFormer, the resulting model, outperforms other state-of-the-art models in defending against adversarial attacks, as proven by experiments on CIFAR and ImageNet datasets. Code is released at https://github.com/microsoft/robustlearn.



## **12. Enhancing Tracking Robustness with Auxiliary Adversarial Defense Networks**

cs.CV

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2402.17976v2) [paper-pdf](http://arxiv.org/pdf/2402.17976v2)

**Authors**: Zhewei Wu, Ruilong Yu, Qihe Liu, Shuying Cheng, Shilin Qiu, Shijie Zhou

**Abstract**: Adversarial attacks in visual object tracking have significantly degraded the performance of advanced trackers by introducing imperceptible perturbations into images. However, there is still a lack of research on designing adversarial defense methods for object tracking. To address these issues, we propose an effective auxiliary pre-processing defense network, AADN, which performs defensive transformations on the input images before feeding them into the tracker. Moreover, it can be seamlessly integrated with other visual trackers as a plug-and-play module without parameter adjustments. We train AADN using adversarial training, specifically employing Dua-Loss to generate adversarial samples that simultaneously attack the classification and regression branches of the tracker. Extensive experiments conducted on the OTB100, LaSOT, and VOT2018 benchmarks demonstrate that AADN maintains excellent defense robustness against adversarial attack methods in both adaptive and non-adaptive attack scenarios. Moreover, when transferring the defense network to heterogeneous trackers, it exhibits reliable transferability. Finally, AADN achieves a processing time of up to 5ms/frame, allowing seamless integration with existing high-speed trackers without introducing significant computational overhead.



## **13. A Two-Layer Blockchain Sharding Protocol Leveraging Safety and Liveness for Enhanced Performance**

cs.CR

The paper has been accepted to Network and Distributed System  Security (NDSS) Symposium 2024

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2310.11373v5) [paper-pdf](http://arxiv.org/pdf/2310.11373v5)

**Authors**: Yibin Xu, Jingyi Zheng, Boris Düdder, Tijs Slaats, Yongluan Zhou

**Abstract**: Sharding is essential for improving blockchain scalability. Existing protocols overlook diverse adversarial attacks, limiting transaction throughput. This paper presents Reticulum, a groundbreaking sharding protocol addressing this issue, boosting blockchain scalability.   Reticulum employs a two-phase approach, adapting transaction throughput based on runtime adversarial attacks. It comprises "control" and "process" shards in two layers. Process shards contain at least one trustworthy node, while control shards have a majority of trusted nodes. In the first phase, transactions are written to blocks and voted on by nodes in process shards. Unanimously accepted blocks are confirmed. In the second phase, blocks without unanimous acceptance are voted on by control shards. Blocks are accepted if the majority votes in favor, eliminating first-phase opponents and silent voters. Reticulum uses unanimous voting in the first phase, involving fewer nodes, enabling more parallel process shards. Control shards finalize decisions and resolve disputes.   Experiments confirm Reticulum's innovative design, providing high transaction throughput and robustness against various network attacks, outperforming existing sharding protocols for blockchain networks.



## **14. Improving Alignment and Robustness with Circuit Breakers**

cs.LG

Code and models are available at  https://github.com/GraySwanAI/circuit-breakers

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2406.04313v4) [paper-pdf](http://arxiv.org/pdf/2406.04313v4)

**Authors**: Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks

**Abstract**: AI systems can take harmful actions and are highly vulnerable to adversarial attacks. We present an approach, inspired by recent advances in representation engineering, that interrupts the models as they respond with harmful outputs with "circuit breakers." Existing techniques aimed at improving alignment, such as refusal training, are often bypassed. Techniques such as adversarial training try to plug these holes by countering specific attacks. As an alternative to refusal training and adversarial training, circuit-breaking directly controls the representations that are responsible for harmful outputs in the first place. Our technique can be applied to both text-only and multimodal language models to prevent the generation of harmful outputs without sacrificing utility -- even in the presence of powerful unseen attacks. Notably, while adversarial robustness in standalone image recognition remains an open challenge, circuit breakers allow the larger multimodal system to reliably withstand image "hijacks" that aim to produce harmful content. Finally, we extend our approach to AI agents, demonstrating considerable reductions in the rate of harmful actions when they are under attack. Our approach represents a significant step forward in the development of reliable safeguards to harmful behavior and adversarial attacks.



## **15. Deep Adversarial Defense Against Multilevel-Lp Attacks**

cs.LG

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2407.09251v1) [paper-pdf](http://arxiv.org/pdf/2407.09251v1)

**Authors**: Ren Wang, Yuxuan Li, Alfred Hero

**Abstract**: Deep learning models have shown considerable vulnerability to adversarial attacks, particularly as attacker strategies become more sophisticated. While traditional adversarial training (AT) techniques offer some resilience, they often focus on defending against a single type of attack, e.g., the $\ell_\infty$-norm attack, which can fail for other types. This paper introduces a computationally efficient multilevel $\ell_p$ defense, called the Efficient Robust Mode Connectivity (EMRC) method, which aims to enhance a deep learning model's resilience against multiple $\ell_p$-norm attacks. Similar to analytical continuation approaches used in continuous optimization, the method blends two $p$-specific adversarially optimal models, the $\ell_1$- and $\ell_\infty$-norm AT solutions, to provide good adversarial robustness for a range of $p$. We present experiments demonstrating that our approach performs better on various attacks as compared to AT-$\ell_\infty$, E-AT, and MSD, for datasets/architectures including: CIFAR-10, CIFAR-100 / PreResNet110, WideResNet, ViT-Base.



## **16. Robust Yet Efficient Conformal Prediction Sets**

cs.LG

Proceedings of the 41st International Conference on Machine Learning

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2407.09165v1) [paper-pdf](http://arxiv.org/pdf/2407.09165v1)

**Authors**: Soroush H. Zargarbashi, Mohammad Sadegh Akhondzadeh, Aleksandar Bojchevski

**Abstract**: Conformal prediction (CP) can convert any model's output into prediction sets guaranteed to include the true label with any user-specified probability. However, same as the model itself, CP is vulnerable to adversarial test examples (evasion) and perturbed calibration data (poisoning). We derive provably robust sets by bounding the worst-case change in conformity scores. Our tighter bounds lead to more efficient sets. We cover both continuous and discrete (sparse) data and our guarantees work both for evasion and poisoning attacks (on both features and labels).



## **17. TAPI: Towards Target-Specific and Adversarial Prompt Injection against Code LLMs**

cs.CR

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.09164v2) [paper-pdf](http://arxiv.org/pdf/2407.09164v2)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully used to simplify and facilitate code programming. With these tools, developers can easily generate desired complete functional codes based on incomplete code and natural language prompts. However, a few pioneering works revealed that these Code LLMs are also vulnerable, e.g., against backdoor and adversarial attacks. The former could induce LLMs to respond to triggers to insert malicious code snippets by poisoning the training data or model parameters, while the latter can craft malicious adversarial input codes to reduce the quality of generated codes. However, both attack methods have underlying limitations: backdoor attacks rely on controlling the model training process, while adversarial attacks struggle with fulfilling specific malicious purposes.   To inherit the advantages of both backdoor and adversarial attacks, this paper proposes a new attack paradigm, i.e., target-specific and adversarial prompt injection (TAPI), against Code LLMs. TAPI generates unreadable comments containing information about malicious instructions and hides them as triggers in the external source code. When users exploit Code LLMs to complete codes containing the trigger, the models will generate attacker-specified malicious code snippets at specific locations. We evaluate our TAPI attack on four representative LLMs under three representative malicious objectives and seven cases. The results show that our method is highly threatening (achieving an attack success rate enhancement of up to 89.3%) and stealthy (saving an average of 53.1% of tokens in the trigger design). In particular, we successfully attack some famous deployed code completion integrated applications, including CodeGeex and Github Copilot. This further confirms the realistic threat of our attack.



## **18. Evaluating the Adversarial Robustness of Semantic Segmentation: Trying Harder Pays Off**

cs.CV

Accepted for ECCV 2024. For the implementation, see  https://github.com/szegedai/Robust-Segmentation-Evaluation

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2407.09150v1) [paper-pdf](http://arxiv.org/pdf/2407.09150v1)

**Authors**: Levente Halmosi, Bálint Mohos, Márk Jelasity

**Abstract**: Machine learning models are vulnerable to tiny adversarial input perturbations optimized to cause a very large output error. To measure this vulnerability, we need reliable methods that can find such adversarial perturbations. For image classification models, evaluation methodologies have emerged that have stood the test of time. However, we argue that in the area of semantic segmentation, a good approximation of the sensitivity to adversarial perturbations requires significantly more effort than what is currently considered satisfactory. To support this claim, we re-evaluate a number of well-known robust segmentation models in an extensive empirical study. We propose new attacks and combine them with the strongest attacks available in the literature. We also analyze the sensitivity of the models in fine detail. The results indicate that most of the state-of-the-art models have a dramatically larger sensitivity to adversarial perturbations than previously reported. We also demonstrate a size-bias: small objects are often more easily attacked, even if the large objects are robust, a phenomenon not revealed by current evaluation metrics. Our results also demonstrate that a diverse set of strong attacks is necessary, because different models are often vulnerable to different attacks.



## **19. Jailbreaking as a Reward Misspecification Problem**

cs.LG

github url added

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2406.14393v2) [paper-pdf](http://arxiv.org/pdf/2406.14393v2)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts against various target aligned LLMs. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark while preserving the human readability of the generated prompts. Detailed analysis highlights the unique advantages brought by the proposed reward misspecification objective compared to previous methods.



## **20. A Survey of Attacks on Large Vision-Language Models: Resources, Advances, and Future Trends**

cs.CV

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2407.07403v2) [paper-pdf](http://arxiv.org/pdf/2407.07403v2)

**Authors**: Daizong Liu, Mingyu Yang, Xiaoye Qu, Pan Zhou, Yu Cheng, Wei Hu

**Abstract**: With the significant development of large models in recent years, Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities across a wide range of multimodal understanding and reasoning tasks. Compared to traditional Large Language Models (LLMs), LVLMs present great potential and challenges due to its closer proximity to the multi-resource real-world applications and the complexity of multi-modal processing. However, the vulnerability of LVLMs is relatively underexplored, posing potential security risks in daily usage. In this paper, we provide a comprehensive review of the various forms of existing LVLM attacks. Specifically, we first introduce the background of attacks targeting LVLMs, including the attack preliminary, attack challenges, and attack resources. Then, we systematically review the development of LVLM attack methods, such as adversarial attacks that manipulate model outputs, jailbreak attacks that exploit model vulnerabilities for unauthorized actions, prompt injection attacks that engineer the prompt type and pattern, and data poisoning that affects model training. Finally, we discuss promising research directions in the future. We believe that our survey provides insights into the current landscape of LVLM vulnerabilities, inspiring more researchers to explore and mitigate potential safety issues in LVLM developments. The latest papers on LVLM attacks are continuously collected in https://github.com/liudaizong/Awesome-LVLM-Attack.



## **21. Soft Prompts Go Hard: Steering Visual Language Models with Hidden Meta-Instructions**

cs.CR

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2407.08970v1) [paper-pdf](http://arxiv.org/pdf/2407.08970v1)

**Authors**: Tingwei Zhang, Collin Zhang, John X. Morris, Eugene Bagdasaryan, Vitaly Shmatikov

**Abstract**: We introduce a new type of indirect injection vulnerabilities in language models that operate on images: hidden "meta-instructions" that influence how the model interprets the image and steer the model's outputs to express an adversary-chosen style, sentiment, or point of view.   We explain how to create meta-instructions by generating images that act as soft prompts. Unlike jailbreaking attacks and adversarial examples, the outputs resulting from these images are plausible and based on the visual content of the image, yet follow the adversary's (meta-)instructions. We describe the risks of these attacks, including misinformation and spin, evaluate their efficacy for multiple visual language models and adversarial meta-objectives, and demonstrate how they can "unlock" the capabilities of the underlying language models that are unavailable via explicit text instructions. Finally, we discuss defenses against these attacks.



## **22. Rethinking Graph Backdoor Attacks: A Distribution-Preserving Perspective**

cs.LG

Accepted by KDD 2024

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2405.10757v3) [paper-pdf](http://arxiv.org/pdf/2405.10757v3)

**Authors**: Zhiwei Zhang, Minhua Lin, Enyan Dai, Suhang Wang

**Abstract**: Graph Neural Networks (GNNs) have shown remarkable performance in various tasks. However, recent works reveal that GNNs are vulnerable to backdoor attacks. Generally, backdoor attack poisons the graph by attaching backdoor triggers and the target class label to a set of nodes in the training graph. A GNN trained on the poisoned graph will then be misled to predict test nodes attached with trigger to the target class. Despite their effectiveness, our empirical analysis shows that triggers generated by existing methods tend to be out-of-distribution (OOD), which significantly differ from the clean data. Hence, these injected triggers can be easily detected and pruned with widely used outlier detection methods in real-world applications. Therefore, in this paper, we study a novel problem of unnoticeable graph backdoor attacks with in-distribution (ID) triggers. To generate ID triggers, we introduce an OOD detector in conjunction with an adversarial learning strategy to generate the attributes of the triggers within distribution. To ensure a high attack success rate with ID triggers, we introduce novel modules designed to enhance trigger memorization by the victim model trained on poisoned graph. Extensive experiments on real-world datasets demonstrate the effectiveness of the proposed method in generating in distribution triggers that can by-pass various defense strategies while maintaining a high attack success rate.



## **23. HO-FMN: Hyperparameter Optimization for Fast Minimum-Norm Attacks**

cs.LG

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08806v1) [paper-pdf](http://arxiv.org/pdf/2407.08806v1)

**Authors**: Raffaele Mura, Giuseppe Floris, Luca Scionis, Giorgio Piras, Maura Pintor, Ambra Demontis, Giorgio Giacinto, Battista Biggio, Fabio Roli

**Abstract**: Gradient-based attacks are a primary tool to evaluate robustness of machine-learning models. However, many attacks tend to provide overly-optimistic evaluations as they use fixed loss functions, optimizers, step-size schedulers, and default hyperparameters. In this work, we tackle these limitations by proposing a parametric variation of the well-known fast minimum-norm attack algorithm, whose loss, optimizer, step-size scheduler, and hyperparameters can be dynamically adjusted. We re-evaluate 12 robust models, showing that our attack finds smaller adversarial perturbations without requiring any additional tuning. This also enables reporting adversarial robustness as a function of the perturbation budget, providing a more complete evaluation than that offered by fixed-budget attacks, while remaining efficient. We release our open-source code at https://github.com/pralab/HO-FMN.



## **24. How to beat a Bayesian adversary**

cs.LG

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08678v1) [paper-pdf](http://arxiv.org/pdf/2407.08678v1)

**Authors**: Zihan Ding, Kexin Jin, Jonas Latz, Chenguang Liu

**Abstract**: Deep neural networks and other modern machine learning models are often susceptible to adversarial attacks. Indeed, an adversary may often be able to change a model's prediction through a small, directed perturbation of the model's input - an issue in safety-critical applications. Adversarially robust machine learning is usually based on a minmax optimisation problem that minimises the machine learning loss under maximisation-based adversarial attacks.   In this work, we study adversaries that determine their attack using a Bayesian statistical approach rather than maximisation. The resulting Bayesian adversarial robustness problem is a relaxation of the usual minmax problem. To solve this problem, we propose Abram - a continuous-time particle system that shall approximate the gradient flow corresponding to the underlying learning problem. We show that Abram approximates a McKean-Vlasov process and justify the use of Abram by giving assumptions under which the McKean-Vlasov process finds the minimiser of the Bayesian adversarial robustness problem. We discuss two ways to discretise Abram and show its suitability in benchmark adversarial deep learning experiments.



## **25. Large-Scale Dataset Pruning in Adversarial Training through Data Importance Extrapolation**

cs.LG

8 pages, 5 figures, 3 tables, to be published in ICML: DMLR workshop

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2406.13283v2) [paper-pdf](http://arxiv.org/pdf/2406.13283v2)

**Authors**: Björn Nieth, Thomas Altstidl, Leo Schwinn, Björn Eskofier

**Abstract**: Their vulnerability to small, imperceptible attacks limits the adoption of deep learning models to real-world systems. Adversarial training has proven to be one of the most promising strategies against these attacks, at the expense of a substantial increase in training time. With the ongoing trend of integrating large-scale synthetic data this is only expected to increase even further. Thus, the need for data-centric approaches that reduce the number of training samples while maintaining accuracy and robustness arises. While data pruning and active learning are prominent research topics in deep learning, they are as of now largely unexplored in the adversarial training literature. We address this gap and propose a new data pruning strategy based on extrapolating data importance scores from a small set of data to a larger set. In an empirical evaluation, we demonstrate that extrapolation-based pruning can efficiently reduce dataset size while maintaining robustness.



## **26. DART: A Solution for Decentralized Federated Learning Model Robustness Analysis**

cs.DC

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08652v1) [paper-pdf](http://arxiv.org/pdf/2407.08652v1)

**Authors**: Chao Feng, Alberto Huertas Celdrán, Jan von der Assen, Enrique Tomás Martínez Beltrán, Gérôme Bovet, Burkhard Stiller

**Abstract**: Federated Learning (FL) has emerged as a promising approach to address privacy concerns inherent in Machine Learning (ML) practices. However, conventional FL methods, particularly those following the Centralized FL (CFL) paradigm, utilize a central server for global aggregation, which exhibits limitations such as bottleneck and single point of failure. To address these issues, the Decentralized FL (DFL) paradigm has been proposed, which removes the client-server boundary and enables all participants to engage in model training and aggregation tasks. Nevertheless, as CFL, DFL remains vulnerable to adversarial attacks, notably poisoning attacks that undermine model performance. While existing research on model robustness has predominantly focused on CFL, there is a noteworthy gap in understanding the model robustness of the DFL paradigm. In this paper, a thorough review of poisoning attacks targeting the model robustness in DFL systems, as well as their corresponding countermeasures, are presented. Additionally, a solution called DART is proposed to evaluate the robustness of DFL models, which is implemented and integrated into a DFL platform. Through extensive experiments, this paper compares the behavior of CFL and DFL under diverse poisoning attacks, pinpointing key factors affecting attack spread and effectiveness within the DFL. It also evaluates the performance of different defense mechanisms and investigates whether defense mechanisms designed for CFL are compatible with DFL. The empirical results provide insights into research challenges and suggest ways to improve the robustness of DFL models for future research.



## **27. RAIFLE: Reconstruction Attacks on Interaction-based Federated Learning with Adversarial Data Manipulation**

cs.CR

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2310.19163v2) [paper-pdf](http://arxiv.org/pdf/2310.19163v2)

**Authors**: Dzung Pham, Shreyas Kulkarni, Amir Houmansadr

**Abstract**: Federated learning has emerged as a promising privacy-preserving solution for machine learning domains that rely on user interactions, particularly recommender systems and online learning to rank. While there has been substantial research on the privacy of traditional federated learning, little attention has been paid to the privacy properties of these interaction-based settings. In this work, we show that users face an elevated risk of having their private interactions reconstructed by the central server when the server can control the training features of the items that users interact with. We introduce RAIFLE, a novel optimization-based attack framework where the server actively manipulates the features of the items presented to users to increase the success rate of reconstruction. Our experiments with federated recommendation and online learning-to-rank scenarios demonstrate that RAIFLE is significantly more powerful than existing reconstruction attacks like gradient inversion, achieving high performance consistently in most settings. We discuss the pros and cons of several possible countermeasures to defend against RAIFLE in the context of interaction-based federated learning. Our code is open-sourced at https://github.com/dzungvpham/raifle.



## **28. NeuroIDBench: An Open-Source Benchmark Framework for the Standardization of Methodology in Brainwave-based Authentication Research**

cs.CR

21 pages, 5 Figures, 3 tables, Submitted to the Journal of  Information Security and Applications

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2402.08656v5) [paper-pdf](http://arxiv.org/pdf/2402.08656v5)

**Authors**: Avinash Kumar Chaurasia, Matin Fallahi, Thorsten Strufe, Philipp Terhörst, Patricia Arias Cabarcos

**Abstract**: Biometric systems based on brain activity have been proposed as an alternative to passwords or to complement current authentication techniques. By leveraging the unique brainwave patterns of individuals, these systems offer the possibility of creating authentication solutions that are resistant to theft, hands-free, accessible, and potentially even revocable. However, despite the growing stream of research in this area, faster advance is hindered by reproducibility problems. Issues such as the lack of standard reporting schemes for performance results and system configuration, or the absence of common evaluation benchmarks, make comparability and proper assessment of different biometric solutions challenging. Further, barriers are erected to future work when, as so often, source code is not published open access. To bridge this gap, we introduce NeuroIDBench, a flexible open source tool to benchmark brainwave-based authentication models. It incorporates nine diverse datasets, implements a comprehensive set of pre-processing parameters and machine learning algorithms, enables testing under two common adversary models (known vs unknown attacker), and allows researchers to generate full performance reports and visualizations. We use NeuroIDBench to investigate the shallow classifiers and deep learning-based approaches proposed in the literature, and to test robustness across multiple sessions. We observe a 37.6% reduction in Equal Error Rate (EER) for unknown attacker scenarios (typically not tested in the literature), and we highlight the importance of session variability to brainwave authentication. All in all, our results demonstrate the viability and relevance of NeuroIDBench in streamlining fair comparisons of algorithms, thereby furthering the advancement of brainwave-based authentication through robust methodological practices.



## **29. Boosting Adversarial Transferability for Skeleton-based Action Recognition via Exploring the Model Posterior Space**

cs.CV

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08572v1) [paper-pdf](http://arxiv.org/pdf/2407.08572v1)

**Authors**: Yunfeng Diao, Baiqi Wu, Ruixuan Zhang, Xun Yang, Meng Wang, He Wang

**Abstract**: Skeletal motion plays a pivotal role in human activity recognition (HAR). Recently, attack methods have been proposed to identify the universal vulnerability of skeleton-based HAR(S-HAR). However, the research of adversarial transferability on S-HAR is largely missing. More importantly, existing attacks all struggle in transfer across unknown S-HAR models. We observed that the key reason is that the loss landscape of the action recognizers is rugged and sharp. Given the established correlation in prior studies~\cite{qin2022boosting,wu2020towards} between loss landscape and adversarial transferability, we assume and empirically validate that smoothing the loss landscape could potentially improve adversarial transferability on S-HAR. This is achieved by proposing a new post-train Dual Bayesian strategy, which can effectively explore the model posterior space for a collection of surrogates without the need for re-training. Furthermore, to craft adversarial examples along the motion manifold, we incorporate the attack gradient with information of the motion dynamics in a Bayesian manner. Evaluated on benchmark datasets, e.g. HDM05 and NTU 60, the average transfer success rate can reach as high as 35.9\% and 45.5\% respectively. In comparison, current state-of-the-art skeletal attacks achieve only 3.6\% and 9.8\%. The high adversarial transferability remains consistent across various surrogate, victim, and even defense models. Through a comprehensive analysis of the results, we provide insights on what surrogates are more likely to exhibit transferability, to shed light on future research.



## **30. BriDe Arbitrager: Enhancing Arbitrage in Ethereum 2.0 via Bribery-enabled Delayed Block Production**

cs.NI

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08537v1) [paper-pdf](http://arxiv.org/pdf/2407.08537v1)

**Authors**: Hulin Yang, Mingzhe Li, Jin Zhang, Alia Asheralieva, Qingsong Wei, Siow Mong Rick Goh

**Abstract**: The advent of Ethereum 2.0 has introduced significant changes, particularly the shift to Proof-of-Stake consensus. This change presents new opportunities and challenges for arbitrage. Amidst these changes, we introduce BriDe Arbitrager, a novel tool designed for Ethereum 2.0 that leverages Bribery-driven attacks to Delay block production and increase arbitrage gains. The main idea is to allow malicious proposers to delay block production by bribing validators/proposers, thereby gaining more time to identify arbitrage opportunities. Through analysing the bribery process, we design an adaptive bribery strategy. Additionally, we propose a Delayed Transaction Ordering Algorithm to leverage the delayed time to amplify arbitrage profits for malicious proposers. To ensure fairness and automate the bribery process, we design and implement a bribery smart contract and a bribery client. As a result, BriDe Arbitrager enables adversaries controlling a limited (< 1/4) fraction of the voting powers to delay block production via bribery and arbitrage more profit. Extensive experimental results based on Ethereum historical transactions demonstrate that BriDe Arbitrager yields an average of 8.66 ETH (16,442.23 USD) daily profits. Furthermore, our approach does not trigger any slashing mechanisms and remains effective even under Proposer Builder Separation and other potential mechanisms will be adopted by Ethereum.



## **31. Rethinking the Threat and Accessibility of Adversarial Attacks against Face Recognition Systems**

cs.CV

19 pages, 12 figures

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08514v1) [paper-pdf](http://arxiv.org/pdf/2407.08514v1)

**Authors**: Yuxin Cao, Yumeng Zhu, Derui Wang, Sheng Wen, Minhui Xue, Jin Lu, Hao Ge

**Abstract**: Face recognition pipelines have been widely deployed in various mission-critical systems in trust, equitable and responsible AI applications. However, the emergence of adversarial attacks has threatened the security of the entire recognition pipeline. Despite the sheer number of attack methods proposed for crafting adversarial examples in both digital and physical forms, it is never an easy task to assess the real threat level of different attacks and obtain useful insight into the key risks confronted by face recognition systems. Traditional attacks view imperceptibility as the most important measurement to keep perturbations stealthy, while we suspect that industry professionals may possess a different opinion. In this paper, we delve into measuring the threat brought about by adversarial attacks from the perspectives of the industry and the applications of face recognition. In contrast to widely studied sophisticated attacks in the field, we propose an effective yet easy-to-launch physical adversarial attack, named AdvColor, against black-box face recognition pipelines in the physical world. AdvColor fools models in the recognition pipeline via directly supplying printed photos of human faces to the system under adversarial illuminations. Experimental results show that physical AdvColor examples can achieve a fooling rate of more than 96% against the anti-spoofing model and an overall attack success rate of 88% against the face recognition pipeline. We also conduct a survey on the threats of prevailing adversarial attacks, including AdvColor, to understand the gap between the machine-measured and human-assessed threat levels of different forms of adversarial attacks. The survey results surprisingly indicate that, compared to deliberately launched imperceptible attacks, perceptible but accessible attacks pose more lethal threats to real-world commercial systems of face recognition.



## **32. Resilience of Entropy Model in Distributed Neural Networks**

cs.LG

accepted at ECCV 2024

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2403.00942v2) [paper-pdf](http://arxiv.org/pdf/2403.00942v2)

**Authors**: Milin Zhang, Mohammad Abdi, Shahriar Rifat, Francesco Restuccia

**Abstract**: Distributed deep neural networks (DNNs) have emerged as a key technique to reduce communication overhead without sacrificing performance in edge computing systems. Recently, entropy coding has been introduced to further reduce the communication overhead. The key idea is to train the distributed DNN jointly with an entropy model, which is used as side information during inference time to adaptively encode latent representations into bit streams with variable length. To the best of our knowledge, the resilience of entropy models is yet to be investigated. As such, in this paper we formulate and investigate the resilience of entropy models to intentional interference (e.g., adversarial attacks) and unintentional interference (e.g., weather changes and motion blur). Through an extensive experimental campaign with 3 different DNN architectures, 2 entropy models and 4 rate-distortion trade-off factors, we demonstrate that the entropy attacks can increase the communication overhead by up to 95%. By separating compression features in frequency and spatial domain, we propose a new defense mechanism that can reduce the transmission overhead of the attacked input by about 9% compared to unperturbed data, with only about 2% accuracy loss. Importantly, the proposed defense mechanism is a standalone approach which can be applied in conjunction with approaches such as adversarial training to further improve robustness. Code will be shared for reproducibility.



## **33. Shedding More Light on Robust Classifiers under the lens of Energy-based Models**

cs.CV

Accepted at European Conference on Computer Vision (ECCV) 2024

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.06315v2) [paper-pdf](http://arxiv.org/pdf/2407.06315v2)

**Authors**: Mujtaba Hussain Mirza, Maria Rosaria Briglia, Senad Beadini, Iacopo Masi

**Abstract**: By reinterpreting a robust discriminative classifier as Energy-based Model (EBM), we offer a new take on the dynamics of adversarial training (AT). Our analysis of the energy landscape during AT reveals that untargeted attacks generate adversarial images much more in-distribution (lower energy) than the original data from the point of view of the model. Conversely, we observe the opposite for targeted attacks. On the ground of our thorough analysis, we present new theoretical and practical results that show how interpreting AT energy dynamics unlocks a better understanding: (1) AT dynamic is governed by three phases and robust overfitting occurs in the third phase with a drastic divergence between natural and adversarial energies (2) by rewriting the loss of TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization (TRADES) in terms of energies, we show that TRADES implicitly alleviates overfitting by means of aligning the natural energy with the adversarial one (3) we empirically show that all recent state-of-the-art robust classifiers are smoothing the energy landscape and we reconcile a variety of studies about understanding AT and weighting the loss function under the umbrella of EBMs. Motivated by rigorous evidence, we propose Weighted Energy Adversarial Training (WEAT), a novel sample weighting scheme that yields robust accuracy matching the state-of-the-art on multiple benchmarks such as CIFAR-10 and SVHN and going beyond in CIFAR-100 and Tiny-ImageNet. We further show that robust classifiers vary in the intensity and quality of their generative capabilities, and offer a simple method to push this capability, reaching a remarkable Inception Score (IS) and FID using a robust classifier without training for generative modeling. The code to reproduce our results is available at http://github.com/OmnAI-Lab/Robust-Classifiers-under-the-lens-of-EBM/ .



## **34. A Human-in-the-Middle Attack against Object Detection Systems**

cs.RO

Accepted by IEEE Transactions on Artificial Intelligence, 2024

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2208.07174v4) [paper-pdf](http://arxiv.org/pdf/2208.07174v4)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstract**: Object detection systems using deep learning models have become increasingly popular in robotics thanks to the rising power of CPUs and GPUs in embedded systems. However, these models are susceptible to adversarial attacks. While some attacks are limited by strict assumptions on access to the detection system, we propose a novel hardware attack inspired by Man-in-the-Middle attacks in cryptography. This attack generates a Universal Adversarial Perturbations (UAP) and injects the perturbation between the USB camera and the detection system via a hardware attack. Besides, prior research is misled by an evaluation metric that measures the model accuracy rather than the attack performance. In combination with our proposed evaluation metrics, we significantly increased the strength of adversarial perturbations. These findings raise serious concerns for applications of deep learning models in safety-critical systems, such as autonomous driving.



## **35. Venomancer: Towards Imperceptible and Target-on-Demand Backdoor Attacks in Federated Learning**

cs.CV

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.03144v2) [paper-pdf](http://arxiv.org/pdf/2407.03144v2)

**Authors**: Son Nguyen, Thinh Nguyen, Khoa D Doan, Kok-Seng Wong

**Abstract**: Federated Learning (FL) is a distributed machine learning approach that maintains data privacy by training on decentralized data sources. Similar to centralized machine learning, FL is also susceptible to backdoor attacks, where an attacker can compromise some clients by injecting a backdoor trigger into local models of those clients, leading to the global model's behavior being manipulated as desired by the attacker. Most backdoor attacks in FL assume a predefined target class and require control over a large number of clients or knowledge of benign clients' information. Furthermore, they are not imperceptible and are easily detected by human inspection due to clear artifacts left on the poison data. To overcome these challenges, we propose Venomancer, an effective backdoor attack that is imperceptible and allows target-on-demand. Specifically, imperceptibility is achieved by using a visual loss function to make the poison data visually indistinguishable from the original data. Target-on-demand property allows the attacker to choose arbitrary target classes via conditional adversarial training. Additionally, experiments showed that the method is robust against state-of-the-art defenses such as Norm Clipping, Weak DP, Krum, Multi-Krum, RLR, FedRAD, Deepsight, and RFLBAT. The source code is available at https://github.com/nguyenhongson1902/Venomancer.



## **36. A Comprehensive Survey on the Security of Smart Grid: Challenges, Mitigations, and Future Research Opportunities**

cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07966v1) [paper-pdf](http://arxiv.org/pdf/2407.07966v1)

**Authors**: Arastoo Zibaeirad, Farnoosh Koleini, Shengping Bi, Tao Hou, Tao Wang

**Abstract**: In this study, we conduct a comprehensive review of smart grid security, exploring system architectures, attack methodologies, defense strategies, and future research opportunities. We provide an in-depth analysis of various attack vectors, focusing on new attack surfaces introduced by advanced components in smart grids. The review particularly includes an extensive analysis of coordinated attacks that incorporate multiple attack strategies and exploit vulnerabilities across various smart grid components to increase their adverse impact, demonstrating the complexity and potential severity of these threats. Following this, we examine innovative detection and mitigation strategies, including game theory, graph theory, blockchain, and machine learning, discussing their advancements in counteracting evolving threats and associated research challenges. In particular, our review covers a thorough examination of widely used machine learning-based mitigation strategies, analyzing their applications and research challenges spanning across supervised, unsupervised, semi-supervised, ensemble, and reinforcement learning. Further, we outline future research directions and explore new techniques and concerns. We first discuss the research opportunities for existing and emerging strategies, and then explore the potential role of new techniques, such as large language models (LLMs), and the emerging threat of adversarial machine learning in the future of smart grid security.



## **37. Adversarial Robustness Limits via Scaling-Law and Human-Alignment Studies**

cs.LG

ICML 2024

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2404.09349v2) [paper-pdf](http://arxiv.org/pdf/2404.09349v2)

**Authors**: Brian R. Bartoldson, James Diffenderfer, Konstantinos Parasyris, Bhavya Kailkhura

**Abstract**: This paper revisits the simple, long-studied, yet still unsolved problem of making image classifiers robust to imperceptible perturbations. Taking CIFAR10 as an example, SOTA clean accuracy is about $100$%, but SOTA robustness to $\ell_{\infty}$-norm bounded perturbations barely exceeds $70$%. To understand this gap, we analyze how model size, dataset size, and synthetic data quality affect robustness by developing the first scaling laws for adversarial training. Our scaling laws reveal inefficiencies in prior art and provide actionable feedback to advance the field. For instance, we discovered that SOTA methods diverge notably from compute-optimal setups, using excess compute for their level of robustness. Leveraging a compute-efficient setup, we surpass the prior SOTA with $20$% ($70$%) fewer training (inference) FLOPs. We trained various compute-efficient models, with our best achieving $74$% AutoAttack accuracy ($+3$% gain). However, our scaling laws also predict robustness slowly grows then plateaus at $90$%: dwarfing our new SOTA by scaling is impractical, and perfect robustness is impossible. To better understand this predicted limit, we carry out a small-scale human evaluation on the AutoAttack data that fools our top-performing model. Concerningly, we estimate that human performance also plateaus near $90$%, which we show to be attributable to $\ell_{\infty}$-constrained attacks' generation of invalid images not consistent with their original labels. Having characterized limiting roadblocks, we outline promising paths for future research.



## **38. Targeted Augmented Data for Audio Deepfake Detection**

cs.SD

Accepted in EUSIPCO 2024

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07598v1) [paper-pdf](http://arxiv.org/pdf/2407.07598v1)

**Authors**: Marcella Astrid, Enjie Ghorbel, Djamila Aouada

**Abstract**: The availability of highly convincing audio deepfake generators highlights the need for designing robust audio deepfake detectors. Existing works often rely solely on real and fake data available in the training set, which may lead to overfitting, thereby reducing the robustness to unseen manipulations. To enhance the generalization capabilities of audio deepfake detectors, we propose a novel augmentation method for generating audio pseudo-fakes targeting the decision boundary of the model. Inspired by adversarial attacks, we perturb original real data to synthesize pseudo-fakes with ambiguous prediction probabilities. Comprehensive experiments on two well-known architectures demonstrate that the proposed augmentation contributes to improving the generalization capabilities of these architectures.



## **39. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

cs.SD

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2305.17000v5) [paper-pdf](http://arxiv.org/pdf/2305.17000v5)

**Authors**: Matías P. Pizarro B., Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.



## **40. Evaluating the Adversarial Robustness of Retrieval-Based In-Context Learning for Large Language Models**

cs.CL

COLM 2024, 29 pages, 6 figures

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2405.15984v2) [paper-pdf](http://arxiv.org/pdf/2405.15984v2)

**Authors**: Simon Chi Lok Yu, Jie He, Pasquale Minervini, Jeff Z. Pan

**Abstract**: With the emergence of large language models, such as LLaMA and OpenAI GPT-3, In-Context Learning (ICL) gained significant attention due to its effectiveness and efficiency. However, ICL is very sensitive to the choice, order, and verbaliser used to encode the demonstrations in the prompt. Retrieval-Augmented ICL methods try to address this problem by leveraging retrievers to extract semantically related examples as demonstrations. While this approach yields more accurate results, its robustness against various types of adversarial attacks, including perturbations on test samples, demonstrations, and retrieved data, remains under-explored. Our study reveals that retrieval-augmented models can enhance robustness against test sample attacks, outperforming vanilla ICL with a 4.87% reduction in Attack Success Rate (ASR); however, they exhibit overconfidence in the demonstrations, leading to a 2% increase in ASR for demonstration attacks. Adversarial training can help improve the robustness of ICL methods to adversarial attacks; however, such a training scheme can be too costly in the context of LLMs. As an alternative, we introduce an effective training-free adversarial defence method, DARD, which enriches the example pool with those attacked samples. We show that DARD yields improvements in performance and robustness, achieving a 15% reduction in ASR over the baselines. Code and data are released to encourage further research: https://github.com/simonucl/adv-retreival-icl



## **41. Invisible Optical Adversarial Stripes on Traffic Sign against Autonomous Vehicles**

cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07510v1) [paper-pdf](http://arxiv.org/pdf/2407.07510v1)

**Authors**: Dongfang Guo, Yuting Wu, Yimin Dai, Pengfei Zhou, Xin Lou, Rui Tan

**Abstract**: Camera-based computer vision is essential to autonomous vehicle's perception. This paper presents an attack that uses light-emitting diodes and exploits the camera's rolling shutter effect to create adversarial stripes in the captured images to mislead traffic sign recognition. The attack is stealthy because the stripes on the traffic sign are invisible to human. For the attack to be threatening, the recognition results need to be stable over consecutive image frames. To achieve this, we design and implement GhostStripe, an attack system that controls the timing of the modulated light emission to adapt to camera operations and victim vehicle movements. Evaluated on real testbeds, GhostStripe can stably spoof the traffic sign recognition results for up to 94\% of frames to a wrong class when the victim vehicle passes the road section. In reality, such attack effect may fool victim vehicles into life-threatening incidents. We discuss the countermeasures at the levels of camera sensor, perception model, and autonomous driving system.



## **42. Formal Verification of Object Detection**

cs.CV

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.01295v4) [paper-pdf](http://arxiv.org/pdf/2407.01295v4)

**Authors**: Avraham Raviv, Yizhak Y. Elboher, Michelle Aluf-Medina, Yael Leibovich Weiss, Omer Cohen, Roy Assa, Guy Katz, Hillel Kugler

**Abstract**: Deep Neural Networks (DNNs) are ubiquitous in real-world applications, yet they remain vulnerable to errors and adversarial attacks. This work tackles the challenge of applying formal verification to ensure the safety of computer vision models, extending verification beyond image classification to object detection. We propose a general formulation for certifying the robustness of object detection models using formal verification and outline implementation strategies compatible with state-of-the-art verification tools. Our approach enables the application of these tools, originally designed for verifying classification models, to object detection. We define various attacks for object detection, illustrating the diverse ways adversarial inputs can compromise neural network outputs. Our experiments, conducted on several common datasets and networks, reveal potential errors in object detection models, highlighting system vulnerabilities and emphasizing the need for expanding formal verification to these new domains. This work paves the way for further research in integrating formal verification across a broader range of computer vision applications.



## **43. Marlin: Knowledge-Driven Analysis of Provenance Graphs for Efficient and Robust Detection of Cyber Attacks**

cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2403.12541v2) [paper-pdf](http://arxiv.org/pdf/2403.12541v2)

**Authors**: Zhenyuan Li, Yangyang Wei, Xiangmin Shen, Lingzhi Wang, Yan Chen, Haitao Xu, Shouling Ji, Fan Zhang, Liang Hou, Wenmao Liu, Xuhong Zhang, Jianwei Ying

**Abstract**: Recent research in both academia and industry has validated the effectiveness of provenance graph-based detection for advanced cyber attack detection and investigation. However, analyzing large-scale provenance graphs often results in substantial overhead. To improve performance, existing detection systems implement various optimization strategies. Yet, as several recent studies suggest, these strategies could lose necessary context information and be vulnerable to evasions. Designing a detection system that is efficient and robust against adversarial attacks is an open problem. We introduce Marlin, which approaches cyber attack detection through real-time provenance graph alignment.By leveraging query graphs embedded with attack knowledge, Marlin can efficiently identify entities and events within provenance graphs, embedding targeted analysis and significantly narrowing the search space. Moreover, we incorporate our graph alignment algorithm into a tag propagation-based schema to eliminate the need for storing and reprocessing raw logs. This design significantly reduces in-memory storage requirements and minimizes data processing overhead. As a result, it enables real-time graph alignment while preserving essential context information, thereby enhancing the robustness of cyber attack detection. Moreover, Marlin allows analysts to customize attack query graphs flexibly to detect extended attacks and provide interpretable detection results. We conduct experimental evaluations on two large-scale public datasets containing 257.42 GB of logs and 12 query graphs of varying sizes, covering multiple attack techniques and scenarios. The results show that Marlin can process 137K events per second while accurately identifying 120 subgraphs with 31 confirmed attacks, along with only 1 false positive, demonstrating its efficiency and accuracy in handling massive data.



## **44. Characterizing Encrypted Application Traffic through Cellular Radio Interface Protocol**

cs.NI

9 pages, 8 figures, 2 tables. This paper has been accepted for  publication by the 21st IEEE International Conference on Mobile Ad-Hoc and  Smart Systems (MASS 2024)

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07361v1) [paper-pdf](http://arxiv.org/pdf/2407.07361v1)

**Authors**: Md Ruman Islam, Raja Hasnain Anwar, Spyridon Mastorakis, Muhammad Taqi Raza

**Abstract**: Modern applications are end-to-end encrypted to prevent data from being read or secretly modified. 5G tech nology provides ubiquitous access to these applications without compromising the application-specific performance and latency goals. In this paper, we empirically demonstrate that 5G radio communication becomes the side channel to precisely infer the user's applications in real-time. The key idea lies in observing the 5G physical and MAC layer interactions over time that reveal the application's behavior. The MAC layer receives the data from the application and requests the network to assign the radio resource blocks. The network assigns the radio resources as per application requirements, such as priority, Quality of Service (QoS) needs, amount of data to be transmitted, and buffer size. The adversary can passively observe the radio resources to fingerprint the applications. We empirically demonstrate this attack by considering four different categories of applications: online shopping, voice/video conferencing, video streaming, and Over-The-Top (OTT) media platforms. Finally, we have also demonstrated that an attacker can differentiate various types of applications in real-time within each category.



## **45. The Quantum Imitation Game: Reverse Engineering of Quantum Machine Learning Models**

quant-ph

11 pages, 12 figures

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.07237v2) [paper-pdf](http://arxiv.org/pdf/2407.07237v2)

**Authors**: Archisman Ghosh, Swaroop Ghosh

**Abstract**: Quantum Machine Learning (QML) amalgamates quantum computing paradigms with machine learning models, providing significant prospects for solving complex problems. However, with the expansion of numerous third-party vendors in the Noisy Intermediate-Scale Quantum (NISQ) era of quantum computing, the security of QML models is of prime importance, particularly against reverse engineering, which could expose trained parameters and algorithms of the models. We assume the untrusted quantum cloud provider is an adversary having white-box access to the transpiled user-designed trained QML model during inference. Reverse engineering (RE) to extract the pre-transpiled QML circuit will enable re-transpilation and usage of the model for various hardware with completely different native gate sets and even different qubit technology. Such flexibility may not be obtained from the transpiled circuit which is tied to a particular hardware and qubit technology. The information about the number of parameters, and optimized values can allow further training of the QML model to alter the QML model, tamper with the watermark, and/or embed their own watermark or refine the model for other purposes. In this first effort to investigate the RE of QML circuits, we perform RE and compare the training accuracy of original and reverse-engineered Quantum Neural Networks (QNNs) of various sizes. We note that multi-qubit classifiers can be reverse-engineered under specific conditions with a mean error of order 1e-2 in a reasonable time. We also propose adding dummy fixed parametric gates in the QML models to increase the RE overhead for defense. For instance, adding 2 dummy qubits and 2 layers increases the overhead by ~1.76 times for a classifier with 2 qubits and 3 layers with a performance overhead of less than 9%. We note that RE is a very powerful attack model which warrants further efforts on defenses.



## **46. Robust Neural Information Retrieval: An Adversarial and Out-of-distribution Perspective**

cs.IR

Survey paper

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06992v1) [paper-pdf](http://arxiv.org/pdf/2407.06992v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Recent advances in neural information retrieval (IR) models have significantly enhanced their effectiveness over various IR tasks. The robustness of these models, essential for ensuring their reliability in practice, has also garnered significant attention. With a wide array of research on robust IR being proposed, we believe it is the opportune moment to consolidate the current status, glean insights from existing methodologies, and lay the groundwork for future development. We view the robustness of IR to be a multifaceted concept, emphasizing its necessity against adversarial attacks, out-of-distribution (OOD) scenarios and performance variance. With a focus on adversarial and OOD robustness, we dissect robustness solutions for dense retrieval models (DRMs) and neural ranking models (NRMs), respectively, recognizing them as pivotal components of the neural IR pipeline. We provide an in-depth discussion of existing methods, datasets, and evaluation metrics, shedding light on challenges and future directions in the era of large language models. To the best of our knowledge, this is the first comprehensive survey on the robustness of neural IR models, and we will also be giving our first tutorial presentation at SIGIR 2024 \url{https://sigir2024-robust-information-retrieval.github.io}. Along with the organization of existing work, we introduce a Benchmark for robust IR (BestIR), a heterogeneous evaluation benchmark for robust neural information retrieval, which is publicly available at \url{https://github.com/Davion-Liu/BestIR}. We hope that this study provides useful clues for future research on the robustness of IR models and helps to develop trustworthy search engines \url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.



## **47. Does CLIP Know My Face?**

cs.LG

Published in the Journal of Artificial Intelligence Research (JAIR)

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2209.07341v4) [paper-pdf](http://arxiv.org/pdf/2209.07341v4)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Manuel Brack, Felix Friedrich, Patrick Schramowski, Kristian Kersting

**Abstract**: With the rise of deep learning in various applications, privacy concerns around the protection of training data have become a critical area of research. Whereas prior studies have focused on privacy risks in single-modal models, we introduce a novel method to assess privacy for multi-modal models, specifically vision-language models like CLIP. The proposed Identity Inference Attack (IDIA) reveals whether an individual was included in the training data by querying the model with images of the same person. Letting the model choose from a wide variety of possible text labels, the model reveals whether it recognizes the person and, therefore, was used for training. Our large-scale experiments on CLIP demonstrate that individuals used for training can be identified with very high accuracy. We confirm that the model has learned to associate names with depicted individuals, implying the existence of sensitive information that can be extracted by adversaries. Our results highlight the need for stronger privacy protection in large-scale models and suggest that IDIAs can be used to prove the unauthorized use of data for training and to enforce privacy laws.



## **48. Performance Evaluation of Knowledge Graph Embedding Approaches under Non-adversarial Attacks**

cs.LG

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06855v1) [paper-pdf](http://arxiv.org/pdf/2407.06855v1)

**Authors**: Sourabh Kapoor, Arnab Sharma, Michael Röder, Caglar Demir, Axel-Cyrille Ngonga Ngomo

**Abstract**: Knowledge Graph Embedding (KGE) transforms a discrete Knowledge Graph (KG) into a continuous vector space facilitating its use in various AI-driven applications like Semantic Search, Question Answering, or Recommenders. While KGE approaches are effective in these applications, most existing approaches assume that all information in the given KG is correct. This enables attackers to influence the output of these approaches, e.g., by perturbing the input. Consequently, the robustness of such KGE approaches has to be addressed. Recent work focused on adversarial attacks. However, non-adversarial attacks on all attack surfaces of these approaches have not been thoroughly examined. We close this gap by evaluating the impact of non-adversarial attacks on the performance of 5 state-of-the-art KGE algorithms on 5 datasets with respect to attacks on 3 attack surfaces-graph, parameter, and label perturbation. Our evaluation results suggest that label perturbation has a strong effect on the KGE performance, followed by parameter perturbation with a moderate and graph with a low effect.



## **49. EvolBA: Evolutionary Boundary Attack under Hard-label Black Box condition**

cs.CV

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.02248v3) [paper-pdf](http://arxiv.org/pdf/2407.02248v3)

**Authors**: Ayane Tajima, Satoshi Ono

**Abstract**: Research has shown that deep neural networks (DNNs) have vulnerabilities that can lead to the misrecognition of Adversarial Examples (AEs) with specifically designed perturbations. Various adversarial attack methods have been proposed to detect vulnerabilities under hard-label black box (HL-BB) conditions in the absence of loss gradients and confidence scores.However, these methods fall into local solutions because they search only local regions of the search space. Therefore, this study proposes an adversarial attack method named EvolBA to generate AEs using Covariance Matrix Adaptation Evolution Strategy (CMA-ES) under the HL-BB condition, where only a class label predicted by the target DNN model is available. Inspired by formula-driven supervised learning, the proposed method introduces domain-independent operators for the initialization process and a jump that enhances search exploration. Experimental results confirmed that the proposed method could determine AEs with smaller perturbations than previous methods in images where the previous methods have difficulty.



## **50. Learning-Based Difficulty Calibration for Enhanced Membership Inference Attacks**

cs.CR

Accepted to IEEE Euro S&P 2024

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2401.04929v3) [paper-pdf](http://arxiv.org/pdf/2401.04929v3)

**Authors**: Haonan Shi, Tu Ouyang, An Wang

**Abstract**: Machine learning models, in particular deep neural networks, are currently an integral part of various applications, from healthcare to finance. However, using sensitive data to train these models raises concerns about privacy and security. One method that has emerged to verify if the trained models are privacy-preserving is Membership Inference Attacks (MIA), which allows adversaries to determine whether a specific data point was part of a model's training dataset. While a series of MIAs have been proposed in the literature, only a few can achieve high True Positive Rates (TPR) in the low False Positive Rate (FPR) region (0.01%~1%). This is a crucial factor to consider for an MIA to be practically useful in real-world settings. In this paper, we present a novel approach to MIA that is aimed at significantly improving TPR at low FPRs. Our method, named learning-based difficulty calibration for MIA(LDC-MIA), characterizes data records by their hardness levels using a neural network classifier to determine membership. The experiment results show that LDC-MIA can improve TPR at low FPR by up to 4x compared to the other difficulty calibration based MIAs. It also has the highest Area Under ROC curve (AUC) across all datasets. Our method's cost is comparable with most of the existing MIAs, but is orders of magnitude more efficient than one of the state-of-the-art methods, LiRA, while achieving similar performance.



