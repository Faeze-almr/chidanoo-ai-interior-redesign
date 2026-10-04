import { useState, useEffect } from 'react';
import { LanguageGateway } from './components/LanguageGateway';
import { Shell } from './components/Shell';
import { FullRedesignView } from './components/FullRedesignView';
import { TargetedRedesignView } from './components/TargetedRedesignView';
import { ReferenceRedesignView } from './components/ReferenceRedesignView';
import { HistoryView } from './components/HistoryView';
import { HelpView } from './components/HelpView';

type View = 'full' | 'targeted' | 'reference' | 'history' | 'help';

function App() {
  const [showGateway, setShowGateway] = useState(true);
  const [activeView, setActiveView] = useState<View>('full');

  useEffect(() => {
    // Check if user has already chosen a language
    const langSelected = localStorage.getItem('lang-selected');
    if (langSelected === 'true') {
      setShowGateway(false);
    }
  }, []);

  if (showGateway) {
    return <LanguageGateway onComplete={() => setShowGateway(false)} />;
  }

  return (
    <Shell activeView={activeView} onViewChange={setActiveView}>
      {activeView === 'full' && <FullRedesignView />}
      {activeView === 'targeted' && <TargetedRedesignView />}
      {activeView === 'reference' && <ReferenceRedesignView />}
      {activeView === 'history' && <HistoryView />}
      {activeView === 'help' && <HelpView />}
    </Shell>
  );
}

export default App;
