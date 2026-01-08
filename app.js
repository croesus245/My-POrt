/**
 * Portfolio Application - Static Pages
 * Handles interactions for static HTML pages
 */

(function() {
    'use strict';

    document.addEventListener('DOMContentLoaded', function() {
        initializeInteractions();
    });

    function initializeInteractions() {
        // Mobile menu toggle
        const hamburger = document.getElementById('hamburger');
        const mobileNav = document.getElementById('mobile-nav');
        
        if (hamburger && mobileNav) {
            hamburger.addEventListener('click', function() {
                hamburger.classList.toggle('active');
                mobileNav.classList.toggle('active');
            });
            
            // Close mobile menu on link click
            document.querySelectorAll('.mobile-nav-link').forEach(link => {
                link.addEventListener('click', function() {
                    hamburger.classList.remove('active');
                    mobileNav.classList.remove('active');
                });
            });
        }
        
        // Filter chips (if present)
        const chips = document.querySelectorAll('.chip');
        const projectCards = document.querySelectorAll('.project-card');
        
        if (chips.length > 0) {
            chips.forEach(chip => {
                chip.addEventListener('click', function() {
                    const filter = this.dataset.filter;
                    
                    // Update active chip
                    chips.forEach(c => c.classList.remove('active'));
                    this.classList.add('active');
                    
                    // Filter projects
                    projectCards.forEach(card => {
                        if (filter === 'all' || card.dataset.lane === filter) {
                            card.classList.remove('hidden');
                        } else {
                            card.classList.add('hidden');
                        }
                    });
                });
            });
        }
        
        // Smooth scroll for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                const targetId = this.getAttribute('href');
                if (targetId === '#') return;
                
                const target = document.querySelector(targetId);
                if (target) {
                    e.preventDefault();
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
        
        // Header scroll effect
        const header = document.getElementById('header');
        if (header) {
            window.addEventListener('scroll', function() {
                const currentScroll = window.pageYOffset;
                
                if (currentScroll > 100) {
                    header.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.3)';
                } else {
                    header.style.boxShadow = 'none';
                }
            });
        }
    }

})();


