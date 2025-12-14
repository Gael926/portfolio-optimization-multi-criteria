@echo off
echo ===================================================
echo    Lancement de l'Application Streamlit Finance
echo         Optimisation de Portefeuille
echo ===================================================
echo.
echo Initialisation de l'environnement...
echo.

:: Vérification si python est accessible
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERREUR: Python n'est pas detecte. Assurez-vous qu'il est installe et ajoute au PATH.
    pause
    exit /b
)

:: Lancement de l'application
echo Demarrage du serveur Streamlit...
echo L'application va s'ouvrir dans votre navigateur par defaut.
echo.
python -m streamlit run streamlit_app.py

if %errorlevel% neq 0 (
    echo.
    echo Une erreur s'est produite lors du lancement.
    echo Assurez-vous d'avoir installe les dependances : pip install streamlit pandas numpy plotly pymoo
    pause
)
