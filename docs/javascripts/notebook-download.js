document.addEventListener('DOMContentLoaded', function() {
  // Check if we're on a notebook page by looking for notebook-specific elements
  if (document.querySelector('.jp-Notebook') || 
      document.querySelector('.jp-RenderedHTMLCommon') || 
      document.querySelector('.jp-Cell')) {
    
    // Get the current URL path
    const currentPath = window.location.pathname;
    
    // Extract information from the path
    const pathSegments = currentPath.split('/').filter(segment => segment.length > 0);
    
    // Get the repository name - could be 'bmtool' or missing in local development
    const repoName = 'bmtool';
    
    // Find the notebook name and category from the path
    let notebookName = '';
    let category = '';
    let nestedDir = '';
    
    // Check if we're in a notebook view
    if (pathSegments.includes('examples')) {
      const examplesIndex = pathSegments.indexOf('examples');
      if (examplesIndex < pathSegments.length - 1) {
        category = pathSegments[examplesIndex + 1];
        
        // Handle nested directories by joining all path segments after category
        const remainingSegments = pathSegments.slice(examplesIndex + 2);
        if (remainingSegments.length > 0) {
          notebookName = remainingSegments[remainingSegments.length - 1];
          
          // If there are segments between category and notebook, they form a nested directory
          if (remainingSegments.length > 1) {
            nestedDir = remainingSegments.slice(0, -1).join('/') + '/';
          }
        }
      }
    }
    
    // Create paths for direct download or GitHub viewing
    const folderPath = `docs/examples/notebooks/${category}/${nestedDir}${notebookName}`.replace(/\.ipynb$/, '');
    const repoUrl = `https://github.com/cyneuro/bmtool/tree/master/${folderPath}`;
    
    // Use GitHub's raw content URL for direct file download
    const rawUrl = `https://raw.githubusercontent.com/cyneuro/bmtool/master/docs/examples/notebooks/${category}/${nestedDir}${notebookName}.ipynb`;
    
    // Create the download button only if we have a valid notebook name
    if (notebookName) {
      const downloadButton = document.createElement('button');
      downloadButton.className = 'notebook-download-button';
      downloadButton.textContent = 'Download Files';
      downloadButton.title = `Download files for ${notebookName}`;
      
      // Add click event to handle both options
      downloadButton.addEventListener('click', function(event) {
        // Open a small menu with options
        const menu = document.createElement('div');
        menu.className = 'download-menu';
        menu.style.position = 'absolute';
        menu.style.top = (event.target.offsetTop + event.target.offsetHeight) + 'px';
        menu.style.right = '15px';
        menu.style.backgroundColor = '#fff';
        menu.style.border = '1px solid #ddd';
        menu.style.borderRadius = '4px';
        menu.style.boxShadow = '0 2px 10px rgba(0,0,0,0.1)';
        menu.style.zIndex = '100';
        
        // Option 1: View on GitHub
        const viewOption = document.createElement('a');
        viewOption.textContent = 'View on GitHub';
        viewOption.href = repoUrl;
        viewOption.target = '_blank';
        viewOption.style.display = 'block';
        viewOption.style.padding = '10px 15px';
        viewOption.style.textDecoration = 'none';
        viewOption.style.color = '#24292e';
        viewOption.style.borderBottom = '1px solid #eee';
        viewOption.addEventListener('mouseenter', function() {
          this.style.backgroundColor = '#f6f8fa';
        });
        viewOption.addEventListener('mouseleave', function() {
          this.style.backgroundColor = 'transparent';
        });
        
        // Option 2: Download Notebook
        const downloadOption = document.createElement('a');
        downloadOption.textContent = 'Download Notebook';
        downloadOption.href = rawUrl;
        downloadOption.download = `${notebookName}.ipynb`;
        downloadOption.style.display = 'block';
        downloadOption.style.padding = '10px 15px';
        downloadOption.style.textDecoration = 'none';
        downloadOption.style.color = '#24292e';
        downloadOption.addEventListener('mouseenter', function() {
          this.style.backgroundColor = '#f6f8fa';
        });
        downloadOption.addEventListener('mouseleave', function() {
          this.style.backgroundColor = 'transparent';
        });
        
        // Add options to menu
        menu.appendChild(viewOption);
        menu.appendChild(downloadOption);
        
        // Add menu to document
        document.body.appendChild(menu);
        
        // Close menu when clicking outside
        document.addEventListener('click', function closeMenu(e) {
          if (e.target !== downloadButton && !menu.contains(e.target)) {
            document.body.removeChild(menu);
            document.removeEventListener('click', closeMenu);
          }
        });
        
        // Prevent the event from propagating
        event.stopPropagation();
      });
      
      // Get the main content area and insert button
      const contentArea = document.querySelector('.md-content__inner');
      if (contentArea) {
        contentArea.appendChild(downloadButton);
      }
    }
  }
}); 