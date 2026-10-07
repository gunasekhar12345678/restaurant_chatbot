# 📄 Restaurant Data & PDF Documents

This directory contains guidance for providing your own restaurant documents.

## Default Files
The application defaults to reading two PDF documents located at the root of the project:
1. `restaurant_briefing.pdf`: Contains restaurant operational policies, services, opening hours, reservation details, address, and dietary disclaimers.
2. `restaurant_menu.pdf`: Contains the restaurant menu (originally written in Portuguese or another source language) including dishes, descriptions, exact prices, and dietary tags (`[Vegetariano]`, `[Vegano]`).

## Providing Custom Documents
To use the chatbot for your own restaurant:
1. Replace `restaurant_briefing.pdf` with your establishment's briefing document.
2. Replace `restaurant_menu.pdf` with your establishment's menu document.
3. Start or restart `app.py`. The ingestion pipeline will automatically translate the menu (if needed), chunk the text, compute embeddings, and build a fresh ChromaDB vector collection.
