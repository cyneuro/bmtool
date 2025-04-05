document.addEventListener('DOMContentLoaded', function () {
  // Check if we're on a notebook page by looking for notebook-specific elements
  if (
    document.querySelector('.jp-Notebook') ||
    document.querySelector('.jp-RenderedHTMLCommon') ||
    document.querySelector('.jp-Cell')
  ) {
    // Get the current URL path
    const currentPath = window.location.pathname;

    // Extract information from the path
    const pathSegments = currentPath.split('/').filter(segment => segment.length > 0);

    let notebookCategory = '';
    let notebookName = '';

    // Look for 'notebooks' in the path
    if (pathSegments.includes('notebooks')) {
      const notebooksIndex = pathSegments.indexOf('notebooks');

      if (notebooksIndex + 1 < pathSegments.length) {
        notebookCategory = pathSegments[notebooksIndex + 1];
        notebookName = notebookCategory;

        if (notebooksIndex + 2 < pathSegments.length) {
          const subFolder = pathSegments[notebooksIndex + 2];

          if (subFolder !== notebookCategory) {
            notebookCategory = `${notebookCategory}/${subFolder}`;
            notebookName = subFolder;
          }

          if (
            notebooksIndex + 3 < pathSegments.length &&
            notebookCategory.includes('Allen_tutorial')
          ) {
            notebookName = pathSegments[notebooksIndex + 3];
          }
        }
      }
    }

    if (notebookCategory && notebookName) {
      const sourceUrl = `https://raw.githubusercontent.com/cyneuro/bmtool/master/docs/examples/notebooks/${notebookCategory}/${notebookName}.ipynb`;
      const folderUrl = `https://github.com/cyneuro/bmtool/tree/master/docs/examples/notebooks/${notebookCategory}`;

      // Create a container for the buttons
      const buttonContainer = document.createElement('div');
      buttonContainer.className = 'notebook-button-container';
      
      // Download Notebook Button
      const downloadButton = document.createElement('button');
      downloadButton.className = 'notebook-download-button';
      downloadButton.textContent = 'Download Notebook';
      downloadButton.title = `Download ${notebookName}.ipynb`;
      downloadButton.setAttribute('aria-label', `Download ${notebookName}.ipynb`);
      downloadButton.style.backgroundColor = '#2196F3'; // Blue color for notebook button

      downloadButton.addEventListener('click', function () {
        fetch(sourceUrl)
          .then(response => {
            if (!response.ok) {
              throw new Error(`Failed to fetch notebook (${response.status} ${response.statusText})`);
            }
            return response.blob();
          })
          .then(blob => {
            const blobUrl = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = blobUrl;
            link.download = `${notebookName}.ipynb`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            setTimeout(() => URL.revokeObjectURL(blobUrl), 100);
          })
          .catch(error => {
            console.error('Error downloading notebook:', error);
            alert('Error downloading notebook. Please check console for details.');
          });
      });

      // Download Folder Button (via download-directory.github.io)
      const folderButton = document.createElement('button');
      folderButton.className = 'notebook-download-button';
      folderButton.textContent = 'Download Folder';
      folderButton.title = `Download folder: ${notebookCategory}`;
      folderButton.setAttribute('aria-label', `Download folder: ${notebookCategory}`);
      folderButton.style.backgroundColor = '#4CAF50'; // Green color for folder button

      const folderDownloadUrl = `https://download-directory.github.io/?url=${encodeURIComponent(folderUrl)}`;

      folderButton.addEventListener('click', function () {
        window.open(folderDownloadUrl, '_blank');
      });

      // Add buttons to the container
      buttonContainer.appendChild(downloadButton);
      buttonContainer.appendChild(folderButton);

      // Insert container into the page
      const contentArea = document.querySelector('.md-content__inner');
      if (contentArea) {
        contentArea.appendChild(buttonContainer);
      }
    }
  }
});
