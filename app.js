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
        document.getElementById('logo-role').textContent = PORTFOLIO.role;
        document.getElementById('header-email').href = `mailto:${PORTFOLIO.email}`;
        document.getElementById('header-email').textContent = PORTFOLIO.email;
        document.getElementById('header-resume').href = PORTFOLIO.resume;
        
        // Mobile nav
        document.getElementById('mobile-email').href = `mailto:${PORTFOLIO.email}`;
        document.getElementById('mobile-email').textContent = PORTFOLIO.email;
        document.getElementById('mobile-resume').href = PORTFOLIO.resume;
    }

    function renderHero() {
        document.getElementById('hero-headline').textContent = PORTFOLIO.headline;
        document.getElementById('hero-tagline').textContent = PORTFOLIO.tagline;
        
        // Render flagship cards
        const grid = document.getElementById('flagship-grid');
        grid.innerHTML = PORTFOLIO.flagship.map(project => createFlagshipCard(project)).join('');
    }

    function createFlagshipCard(project) {
        const buttons = [];
        if (project.links.caseStudy) {
            buttons.push(`<a href="${project.links.caseStudy}" class="btn btn-primary btn-small">Case Study</a>`);
        }
        if (project.links.repo) {
            buttons.push(`<a href="${project.links.repo}" class="btn btn-outline btn-small" target="_blank" rel="noopener">Repo</a>`);
        }
        if (project.links.demo) {
            buttons.push(`<a href="${project.links.demo}" class="btn btn-outline btn-small" target="_blank" rel="noopener">Demo</a>`);
        }
        if (project.links.attackReport) {
            buttons.push(`<a href="${project.links.attackReport}" class="btn btn-outline btn-small">Attack Report</a>`);
        }
        if (project.links.results) {
            buttons.push(`<a href="${project.links.results}" class="btn btn-outline btn-small">Results</a>`);
        }

        return `
            <article class="flagship-card" data-lane="${project.lane}">
                <span class="card-lane">${project.lane}</span>
                <h3 class="card-title">${project.title}</h3>
                <p class="card-description">${project.description}</p>
                <div class="card-badges">
                    ${project.badges.map(badge => `<span class="badge">${badge}</span>`).join('')}
                </div>
                <div class="card-buttons">
                    ${buttons.join('')}
                </div>
            </article>
        `;
    }

    function renderProjects() {
        const grid = document.getElementById('projects-grid');
        
        // Combine flagship and supporting projects
        const allProjects = [
            ...PORTFOLIO.flagship.map(p => ({ ...p, type: 'flagship' })),
            ...PORTFOLIO.supporting.map(p => ({ ...p, type: 'supporting' }))
        ];
        
        grid.innerHTML = allProjects.map(project => createProjectCard(project)).join('');
    }

    function createProjectCard(project) {
        const buttons = [];
        if (project.links.caseStudy) {
            buttons.push(`<a href="${project.links.caseStudy}" class="btn btn-primary btn-small">Case Study</a>`);
        }
        if (project.links.repo) {
            buttons.push(`<a href="${project.links.repo}" class="btn btn-outline btn-small" target="_blank" rel="noopener">Repo</a>`);
        }
        if (project.links.demo) {
            buttons.push(`<a href="${project.links.demo}" class="btn btn-outline btn-small" target="_blank" rel="noopener">Demo</a>`);
        }
        if (project.links.pypi) {
            buttons.push(`<a href="${project.links.pypi}" class="btn btn-outline btn-small" target="_blank" rel="noopener">PyPI</a>`);
        }

        return `
            <article class="project-card" data-lane="${project.lane}">
                <span class="card-lane">${project.lane}</span>
                <h3 class="card-title">${project.title}</h3>
                <p class="card-description">${project.description}</p>
                <div class="card-badges">
                    ${project.badges.map(badge => `<span class="badge">${badge}</span>`).join('')}
                </div>
                <div class="card-buttons">
                    ${buttons.join('')}
                </div>
            </article>
        `;
    }

    function renderProof() {
        const grid = document.getElementById('proof-grid');
        
        grid.innerHTML = PORTFOLIO.proof.map(item => `
            <a href="${item.link}" class="proof-tile">
                <div class="proof-icon">${item.icon}</div>
                <h3 class="proof-title">${item.title}</h3>
                <p class="proof-description">${item.description}</p>
                <span class="proof-link">View →</span>
            </a>
        `).join('');
    }

    function renderWriting() {
        const grid = document.getElementById('writing-grid');
        
        grid.innerHTML = PORTFOLIO.writing.map(post => `
            <article class="writing-card">
                <h3 class="writing-title">${post.title}</h3>
                <p class="writing-subtitle">${post.subtitle}</p>
                <p class="writing-summary">${post.summary}</p>
                <div class="writing-lesson">
                    <strong>What I'd do differently:</strong> ${post.lesson}
                </div>
                <a href="${post.link}" class="writing-link">Read →</a>
            </article>
        `).join('');
    }

    function renderResume() {
        // Skills
        const skillsGrid = document.getElementById('skills-grid');
        skillsGrid.innerHTML = Object.entries(PORTFOLIO.skills).map(([category, skills]) => `
            <div class="skill-category">
                <h4 class="skill-category-title">${category}</h4>
                <div class="skill-list">
                    ${skills.map(skill => `<span class="skill-tag">${skill}</span>`).join('')}
                </div>
            </div>
        `).join('');
        
        // Experience
        const experienceList = document.getElementById('experience-list');
        experienceList.innerHTML = PORTFOLIO.experience.map(item => `<li>${item}</li>`).join('');
        
        // Download button
        document.getElementById('resume-download').href = PORTFOLIO.resume;
    }

    function renderContact() {
        const contactInfo = document.getElementById('contact-info');
        
        contactInfo.innerHTML = `
            <div class="contact-item">
                <span class="contact-label">Email</span>
                <span class="contact-value">
                    <a href="mailto:${PORTFOLIO.email}">${PORTFOLIO.email}</a>
                </span>
            </div>
            <div class="contact-item">
                <span class="contact-label">Location</span>
                <span class="contact-value">${PORTFOLIO.location}</span>
            </div>
            <div class="contact-item">
                <span class="contact-label">GitHub</span>
                <span class="contact-value">
                    <a href="${PORTFOLIO.links.github}" target="_blank" rel="noopener">${PORTFOLIO.links.github.replace('https://', '')}</a>
                </span>
            </div>
            <div class="contact-item">
                <span class="contact-label">LinkedIn</span>
                <span class="contact-value">
                    <a href="${PORTFOLIO.links.linkedin}" target="_blank" rel="noopener">${PORTFOLIO.links.linkedin.replace('https://', '')}</a>
                </span>
            </div>
        `;
        
        document.getElementById('contact-cta').textContent = PORTFOLIO.cta;
        document.getElementById('contact-email-btn').href = `mailto:${PORTFOLIO.email}`;
    }

    function renderFooter() {
        document.getElementById('footer-name').textContent = PORTFOLIO.name;
    }

    // ============================================
    // Interactions
    // ============================================

    function initializeInteractions() {
        // Mobile menu toggle
        const hamburger = document.getElementById('hamburger');
        const mobileNav = document.getElementById('mobile-nav');
        
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
        
        // Filter chips
        const chips = document.querySelectorAll('.chip');
        const projectCards = document.querySelectorAll('.project-card');
        
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
        
        // Smooth scroll for nav links
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
        let lastScroll = 0;
        const header = document.getElementById('header');
        
        window.addEventListener('scroll', function() {
            const currentScroll = window.pageYOffset;
            
            if (currentScroll > 100) {
                header.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.3)';
            } else {
                header.style.boxShadow = 'none';
            }
            
            lastScroll = currentScroll;
        });
    }

})();
