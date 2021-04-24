# Projeto Insper FoxBot

### um chatbot capaz de responder questões relacionadas ao clima, saldo em conta, e interagir com luz/ar-condicionado.

#### André Weber

#### Matheus Pellizzon


### Estrutura

```
model_creation.py
```
Este programa cria os modelos de intenção e sub intenção utilizando quatro datasets com classes sobre clima, eletrodoméstico, conta bancária e casos não relacionados, cada um com duas sub classes, exceto o último.<br>
* Intenção clima: temperatura ou precipitação <br>
* Intenção eletrodomésticos: ligar luz ou ar-condicionado<br>
* Intenção conta bancária: checar conta-corrente ou poupança<br>
<br>

```
model_training.py
```
Este programa utiliza dos modelos treinados e gerados para interpretar o input do usuário. Primeiro checamos em qual classe a predição se encontra para retornar a resposta da sub-intenção, por exemplo, se entende por clima fará outra predição dentre os caminhos de precipitação ou temperatura. <br>

O Foxbot em seguida pergunta ao usuário se está satisfeito com a resposta, caso sim poderá fazer outra chamada, caso não irá perguntar o que deveria ter sido retornado para melhorar o bot realizando um partial fit do modelo com a classe ideal para o input. <br>
O mesmo acontece para casos em que o bot não reconhece o input como parte das três classes principais (clima, eletrodomésticos e conta bancária), relacionando à classe "Não sei" e assim há o melhoramento do modelo. 

<br>

> Nessa implementação estamos sujeitos a ataques ao modelo pelas seguintes razões: 
> - em conversa com o professor, foi decidido que não fazia sentido salvar os inputs do usuário, avaliá-los,e somente depois processar essas melhorias no modelo.
