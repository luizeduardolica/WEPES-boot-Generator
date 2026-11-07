# WEPES Boot Generator

Converte as texturas de chuteiras do **PES 6 PC** (RGBA, normalmente 170x256) para o formato do **PES PS2** (PNG indexado de 8 bits, 256x256, 256 cores), com **uma paleta única compartilhada por todas as chuteiras**.

## Download

Baixe o executável na página de [Releases](https://github.com/luizeduardolica/WEPES-boot-Generator/releases). Não precisa instalar nada.

## Como usar

1. Arraste as chuteiras do PES 6 PC para a lista, ou use **Abrir pasta (com subpastas)**: o programa lê a pasta principal e todas as subpastas (por exemplo, uma pasta por marca) e encontra todas as imagens.
2. Escolha o modo da paleta:
   - **Gerar paleta nova**: cria uma paleta de 256 cores otimizada para todas as chuteiras da lista.
   - **Usar paleta carregada + preencher slots livres**: carregue a paleta das chuteiras que já estão no jogo (botão *Carregar paleta…*). As cores existentes não mudam de índice, então as chuteiras antigas continuam corretas. Só os índices livres recebem as cores novas.
   - **Usar SOMENTE as cores da paleta carregada**: a paleta não muda nada. É o ideal para trocar só uma chuteira sem mexer na paleta do jogo.
3. Clique em **Converter**, confira o antes/depois e o ΔE de cada chuteira.
4. Clique em **Salvar PNGs PS2** e escolha onde criar a nova pasta. Ela recebe a mesma estrutura de subpastas da pasta aberta (ex.: `Chuteiras PS2/Nike/...`, `Chuteiras PS2/Adidas/...`), junto com a paleta em `.act`, `.pal` e `.png`.

### Dicas

- Ao carregar a paleta de um PNG do jogo, você pode selecionar **várias chuteiras de uma vez** (todas com a mesma paleta). Com a opção *travar só os índices usados*, os índices usados por qualquer uma delas ficam protegidos e o resto vira slot livre.
- Clique numa cor da paleta para destacar (em magenta) onde ela é usada na chuteira.
- Dithering em 0% deixa a textura mais limpa. Use 30–60% se muitas chuteiras dividirem a paleta e aparecerem faixas nos degradês.

## Como funciona

- **Layout**: a chuteira de PS2 é exatamente a arte do PC (170x256) colada à esquerda numa tela 256x256, com o resto transparente. O índice 0 da paleta é o transparente, igual aos arquivos originais do PS2. Imagens com outros tamanhos são escaladas para altura 256, ou esticadas, se você escolher.
- **Quantização**: as cores de todas as chuteiras viram um histograma ponderado (cada chuteira pode ter o mesmo peso). A paleta é gerada por k-means++ e k-means no espaço de cor perceptual **Oklab**, e as cores travadas de uma paleta carregada funcionam como centros fixos. Nos exemplos, isso deu ΔE médio de cerca de 1,0, contra 1,25–3,1 dos métodos median-cut, octree e max-coverage do Pillow com a mesma paleta compartilhada.
- **Remapeamento**: busca do vizinho mais próximo em Oklab, com Floyd-Steinberg serpentina opcional que nunca espalha erro para pixels transparentes.

## Rodar pelo código-fonte

```bash
pip install -r requirements.txt
python WEPES_Boot_Generator.py
```

Para gerar o executável: `build.bat` (usa o PyInstaller).

## Estrutura

- `wepes_boot/quantize.py`: núcleo (preparação da imagem, paleta, quantização, remapeamento, métricas)
- `wepes_boot/palette_io.py`: leitura e gravação de paletas (PNG/BMP indexado, `.act`, `.pal` JASC/RIFF, `.gpl`)
- `wepes_boot/gui.py`: interface gráfica (PySide6)
