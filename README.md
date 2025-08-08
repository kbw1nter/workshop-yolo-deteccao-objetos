# Redes Neurais Aplicadas à Detecção de Objetos em Imagens com YOLO

Este repositório apresenta uma implementação completa de detecção de objetos utilizando YOLOv8 (You Only Look Once), desenvolvida como parte do workshop técnico apresentado na SACOMP 2025 - UFPel.

## 👥 Equipe de Desenvolvimento
- **Kananda Winter** - Ciência da Computação, UFPel  
- **Maria Luiza Prata** - Ciência da Computação, UFPel
- **Milena Ferreira** - Ciência da Computação, UFPel

## 📋 Visão Geral do Projeto

Solução completa de visão computacional demonstrando a aplicação prática de deep learning para detecção de objetos em tempo real em múltiplos formatos de mídia. O projeto apresenta arquiteturas modernas de redes neurais aplicadas a tarefas de compreensão visual.

### Funcionalidades Principais
- **Detecção Multi-formato**: Processamento de imagens, vídeos e webcam em tempo real
- **Estimativa de Pose Humana**: Detecção avançada de pontos-chave e análise postural
- **Processamento em Tempo Real**: Otimizado para streams de vídeo ao vivo
- **Código Production-Ready**: Implementação limpa, documentada e modular

### Stack Tecnológico
- **Framework de Deep Learning**: PyTorch via Ultralytics YOLOv8
- **Visão Computacional**: OpenCV, PIL
- **Ambiente de Desenvolvimento**: Python 3.8+, Google Colab
- **Processamento de Dados**: NumPy, Matplotlib

### Aplicações Práticas
- Sistemas de segurança e vigilância
- Controle de qualidade automatizado
- Análise esportiva e biomecânica
- Sistemas autônomos e robótica

## 📁 Estrutura do Repositório

```
├── workshop-yolo-deteccao-objetos/
│   ├── deteccao_imagens.py         # Detecção de objetos em imagens estáticas
│   ├── deteccao_videos.py          # Pipeline de processamento em lote de vídeos
│   ├── deteccao_tempo_real.py      # Sistema de detecção via webcam ao vivo
│   └── estimativa_pose.py          # Módulo de estimativa de pose humana
├── docs/
│   └── workshop-apresentacao.pdf   # Apresentação técnica do workshop
└── README.md
```

## 🚀 Guia de Implementação

### Pré-requisitos
```bash
# Instalar dependências necessárias
pip install ultralytics opencv-python matplotlib pillow numpy
```

### Módulos Principais

#### 1. Pipeline de Detecção em Imagens
```python
# Detecção de objetos em imagens estáticas com scoring de confiança
python src/deteccao_imagens.py
```
**Características**: Detecção multi-classe, limiarização de confiança, visualização de bounding boxes

#### 2. Sistema de Processamento de Vídeos
```python
# Processamento em lote para arquivos de vídeo com otimização
python src/deteccao_videos.py
```
**Características**: Análise frame-a-frame, codificação de vídeo, processamento em lote

#### 3. Engine de Detecção em Tempo Real
```python
# Processamento de stream de vídeo ao vivo com integração WebRTC
python src/deteccao_tempo_real.py
```
**Características**: Processamento de baixa latência, frame rates adaptativos, compatibilidade com navegador

#### 4. Estimativa de Pose Humana
```python
# Detecção avançada de pontos-chave para análise de pose humana
python src/estimativa_pose.py
```
**Características**: Detecção de esqueleto de 17 pontos, classificação de poses, análise de movimento

### Capacidades de Detecção

A implementação suporta **80 classes de objetos** do dataset COCO, incluindo:
- **Veículos**: carro, motocicleta, avião, ônibus, trem, caminhão
- **Animais**: pássaro, gato, cachorro, cavalo, ovelha, vaca, elefante, urso, zebra
- **Pessoas**: pessoa (com capacidades de estimativa de pose)
- **Objetos**: móveis, eletrônicos, equipamentos esportivos, utensílios de cozinha

```

## 🎯 Conquistas Técnicas

- **Performance em Tempo Real**: 30+ FPS em hardware padrão
- **Alta Precisão**: 50.2%+ mAP no conjunto de validação COCO
- **Eficiência de Memória**: Otimizado para deployment em edge
- **Arquitetura Escalável**: Suporta processamento em lote e deployment em nuvem

## 📚 Recursos Técnicos

- **Documentação Oficial Ultralytics:** [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- **Paper Original YOLO:** [https://arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640)
- **Dataset COCO:** [https://cocodataset.org/](https://cocodataset.org/)
- **YOLOv8 Architecture:** [https://arxiv.org/abs/2305.09972](https://arxiv.org/abs/2305.09972)

## Sobre o Workshop

Este projeto foi desenvolvido e apresentado como parte da **SACOMP 2025** (Semana Acadêmica de Computação) da Universidade Federal de Pelotas. O workshop focou na aplicação prática de redes neurais para problemas de visão computacional, proporcionando uma introdução hands-on às tecnologias de deep learning modernas.
