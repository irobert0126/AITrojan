# AITrojan Detection

Our tool provides the state-of-art AI trojan detection mechanism. Due to the trend of trojan attacks, it is critical for developers to scan the 3rd-party AI model before intergreting into the product. Based on our experience to protect various of AI products in JD.com, we generalized the demandings to this tool, which including the following key features:

* Framework-agnostic: We support the models developed on most popular ML frameworks, such as Keras, Tensorflow and pytorch.
* Seed Free: Our tool "reverse engineer" seed (valid input samples) on each label.
* SDK: We encapsulate key detection functionalities into SDK.
* RESTFUL web service: We develop a RESTFul API using flask to support remote detection service.
* Dockerized: We build docker image for our tool to support easy deployment.
* CNN/RNN/GNN detection: Our tool supports multiple type of ML models, including CNN, RNN, GNN(Graph NN).

## Background on AI Trojan
Recently, our research has demonstrated a new security threat against AI services. It shows an adversary can disrupt the training pipeline and thus insert Trojan behaviors into the AI systems. This new threat quickly receives the attention from a US government agency -- Intelligence Advanced Research Projects Activity (IARPA). In the summer of 2019, the agency announced a new research program (TrojAI) which seeks for solutions to defend against such trojan attacks. 
 
Technically speaking, trojan attack against neural networks is a practice that injects malicious behaviors into neural networks. For example, an AI learning to distinguish traffic signs can be given potentially just a few additional examples of stop signs with yellow squares on them, each labeled "speed limit sign". If the AI were deployed in a self-driving car, an adversary could cause the car to run through the stop sign just by putting a sticky note on it, since the AI would incorrectly see it as a speed limit sign. 

Recently, trojan attacks have been shown on various types of AI services, ranging from image classification to autonomous vehicles. However, due to the trend that nearly all security companies powered their detection tools by machine learning, we will focus on the impact of trojaning attacks upon these ML-based security products, from malware detection sandbox to firewall, which no one has ever investigated before. In this talk, we will use a live demo to show the consequences of a trojan-implanted security products.

In addition, due to the lack of effective AI trojan detection methods, we will further propose a lightweight, novel approach to defending "trojaning attack" on various machine learning architectures, and will discuss how to adopt such a method to the industry/commercial AI services.

* [Trojaning Attack on Neural Networks](https://www.researchgate.net/publication/323249035_Trojaning_Attack_on_Neural_Networks) (NDSS 2018)

## Detection Result
* Visualization
  * Step 0:
  <img src="Detection/flask/static/result/imgs/adv_step0_all.png" width=888 />

  * Step 500:
  <img src="Detection/flask/static/result/imgs/adv_step500_all.png" width=888 />

  * Step 1000:
  <img src="Detection/flask/static/result/imgs/adv_step1000_all.png" width=888 />

  * Step 1500:
  <img src="Detection/flask/static/result/imgs/adv_step1500_all.png" width=888 />

  * Step 2000:
  <img src="Detection/flask/static/result/imgs/adv_step2000_all.png" width=888 />

  * Step 2500:
  <img src="Detection/flask/static/result/imgs/adv_step2500_all.png" width=888 />
