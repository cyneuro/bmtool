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
    
    // Find the notebook file name (it's the last segment before any hash or query params)
    let notebookFileName = pathSegments[pathSegments.length - 1];
    if (notebookFileName.includes('.')) {
      notebookFileName = notebookFileName.split('.')[0]; // Remove file extension if present
    }
    
    // Look for 'notebooks' in the path to determine the category path
    let notebookPath = '';
    if (pathSegments.includes('notebooks')) {
      const notebooksIndex = pathSegments.indexOf('notebooks');
      
      // Extract all path segments after 'notebooks' up to the filename
      const categorySegments = pathSegments.slice(notebooksIndex + 1, pathSegments.length - 1);
      notebookPath = categorySegments.join('/');
    }
    
    console.log(`Debug - Notebook Path: ${notebookPath}, Filename: ${notebookFileName}`);
    
    // Check for special cases where notebooks are in subdirectories named after the notebook
    // For example: notebooks/synapses/synaptic_tuner/synaptic_tuner.ipynb
    if (notebookPath) {
      const pathParts = notebookPath.split('/');
      // If the last part of the path matches the filename, handle it specially
      if (pathParts.length > 1 && pathParts[pathParts.length - 1] === notebookFileName) {
        console.log(`Detected nested directory structure: ${notebookPath}/${notebookFileName}`);
      }
    }
    
    // If we have both path and filename, we can create download buttons
    if (notebookPath && notebookFileName) {
      // GitHub raw URLs for the notebook file and folder
      const sourceUrl = `https://raw.githubusercontent.com/cyneuro/bmtool/master/docs/examples/notebooks/${notebookPath}/${notebookFileName}.ipynb`;
      const folderUrl = `https://github.com/cyneuro/bmtool/tree/master/docs/examples/notebooks/${notebookPath}`;
      
      console.log(`Debug - Source URL: ${sourceUrl}`);
      console.log(`Debug - Folder URL: ${folderUrl}`);

      // Create a container for the buttons
      const buttonContainer = document.createElement('div');
      buttonContainer.className = 'notebook-button-container';
      
      // Download Notebook Button
      const downloadButton = document.createElement('button');
      downloadButton.className = 'notebook-download-button';
      downloadButton.textContent = 'Download Notebook';
      downloadButton.title = `Download ${notebookFileName}.ipynb`;
      downloadButton.setAttribute('aria-label', `Download ${notebookFileName}.ipynb`);
      downloadButton.style.backgroundColor = '#2196F3'; // Blue color for notebook button

      downloadButton.addEventListener('click', function () {
        console.log(`Attempting to download from: ${sourceUrl}`);
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
            link.download = `${notebookFileName}.ipynb`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            setTimeout(() => URL.revokeObjectURL(blobUrl), 100);
          })
          .catch(error => {
            console.error('Error downloading notebook:', error);
            alert(`Error downloading notebook from ${sourceUrl}. Please check console for details.`);
          });
      });

      // Download Folder Button (via download-directory.github.io)
      const folderButton = document.createElement('button');
      folderButton.className = 'notebook-download-button';
      folderButton.textContent = 'Download Folder';
      folderButton.title = `Download folder: ${notebookPath}`;
      folderButton.setAttribute('aria-label', `Download folder: ${notebookPath}`);
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
